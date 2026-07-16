"""Tests for the v2 judged evaluation run trigger (`POST /api/v2/evaluations`).

Covers the ground-truth slice of the three-metric SRD at the route boundary:
a v2 fast run is always a judged run (FR-9 no-flag), batch mode routes to the v1
batch path and never judges, and the v1 trigger stays cosine-only (FR-18). Judging
is system-config only — there is no per-run judge_config in the v2 body. External
dispatch boundaries (Langfuse, dataset fetch, chunk enqueue) are mocked; the DB is
real.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Config, EvaluationDataset, EvaluationRun
from app.models.evaluation import RunModeEnum
from app.models.llm.request import ConfigBlob, KaapiCompletionConfig
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)
from app.tests.utils.utils import random_lower_string

V2_EVALS = f"{settings.API_V2_STR}/evaluations"


def _make_dataset(*, db: Session, user_api_key: TestAuthContext) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        original_items_count=3,
        duplication_factor=1,
    )


def _make_text_config(db: Session, project_id: int) -> Config:
    blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={"model": "gpt-4o-fast-eval-test", "temperature": 0.7},
        )
    )
    return create_test_config(
        db=db, project_id=project_id, use_kaapi_schema=True, config_blob=blob
    )


def _dataset_item(item_id: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "input": {"question": "Q"},
        "expected_output": {"answer": "A"},
        "metadata": {"question_id": item_id},
    }


@pytest.fixture
def _patch_dispatch() -> Iterator[MagicMock]:
    """Stub the synchronous request-path boundaries of validate_and_start.

    Same boundaries the v1 fast route stubs — Langfuse client, dataset fetch, and
    the chunk enqueue — so a v2 run reaches dispatch without real I/O. Yields the
    chunk-enqueue mock (call count == number of dispatched chunks).
    """
    with (
        patch("app.services.evaluations.fast.get_langfuse_client"),
        patch(
            "app.services.evaluations.fast.fetch_dataset_items",
            return_value=[_dataset_item(f"item-{i}") for i in range(3)],
        ),
        patch(
            "app.services.evaluations.fast.start_fast_evaluation_chunk",
            return_value="fake-task-id",
        ) as mock_start_chunk,
    ):
        yield mock_start_chunk


class TestV2JudgedRunTrigger:
    def test_fast_run_marks_judge_run_and_dispatches(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """FR-9: a v2 fast run is always a judged run — no flag, dispatched."""
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        resp = client.post(
            V2_EVALS,
            json={
                "experiment_name": "v2-judged",
                "dataset_id": dataset.id,
                "config_id": str(config.id),
                "config_version": 1,
                "run_mode": "fast",
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["run_mode"] == "fast"
        assert body["status"] == "processing"
        assert body["is_judge_run"] is True
        _patch_dispatch.assert_called_once()

        run = db.get(EvaluationRun, body["id"])
        assert run is not None
        assert run.is_judge_run is True

    def test_run_mode_defaults_to_fast_and_judges(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """The v2 trigger defaults run_mode to fast, so the run judges by default."""
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        resp = client.post(
            V2_EVALS,
            json={
                "experiment_name": "v2-default-mode",
                "dataset_id": dataset.id,
                "config_id": str(config.id),
                "config_version": 1,
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["run_mode"] == "fast"
        assert body["is_judge_run"] is True


class TestV2BatchMode:
    def test_batch_mode_routes_to_batch_and_is_not_judged(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """run_mode='batch' takes the v1 batch branch, which never judges and never
        dispatches the fast judge pipeline."""
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        batch_run = EvaluationRun(
            run_name=f"v2-batch-{random_lower_string()}",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            status="processing",
            run_mode=RunModeEnum.BATCH.value,
            total_items=3,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(batch_run)
        db.commit()
        db.refresh(batch_run)

        # The batch subsystem submits provider batch jobs — mock it at the route boundary.
        with patch(
            "app.api.routes.evaluations.evaluation_v2.validate_and_start_batch_evaluation",
            return_value=batch_run,
        ):
            resp = client.post(
                V2_EVALS,
                json={
                    "experiment_name": batch_run.run_name,
                    "dataset_id": dataset.id,
                    "config_id": str(config.id),
                    "config_version": 1,
                    "run_mode": "batch",
                },
                headers=user_api_key_header,
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["run_mode"] == "batch"
        # Batch never takes the judged fast path.
        _patch_dispatch.assert_not_called()
        run = db.get(EvaluationRun, body["id"])
        assert not run.is_judge_run


class TestV1TriggerUnchanged:
    def test_v1_fast_run_is_not_a_judge_run(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """FR-18: the v1 trigger produces a cosine-only run — never a judged one."""
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        resp = client.post(
            f"{settings.API_V1_STR}/evaluations",
            json={
                "experiment_name": f"v1-plain-{random_lower_string()}",
                "dataset_id": dataset.id,
                "config_id": str(config.id),
                "config_version": 1,
                "run_mode": "fast",
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        run = db.get(EvaluationRun, body["id"])
        assert not run.is_judge_run
        assert run.per_item_ground_truth is None
