"""Tests for the v2 judged evaluation run trigger (`POST /api/v2/evaluations`).

Covers the ground-truth slice of the three-metric SRD at the route boundary:
a v2 run is always fast and always judged (FR-9 no-flag; there is no run_mode —
batch is deferred), and the v1 trigger stays cosine-only (FR-18). Judging is
system-config only — there is no per-run judge_config in the v2 body. External
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
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["status"] == "processing"
        assert body["is_judge_run"] is True
        _patch_dispatch.assert_called_once()

        run = db.get(EvaluationRun, body["id"])
        assert run is not None
        assert run.is_judge_run is True
        assert run.callback_url is None  # no callback_url sent -> stays NULL

    def test_v2_run_is_always_fast_and_judged(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """A v2 run carries no run_mode; it is always created fast and judged."""
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


class TestV2CallbackUrl:
    def test_valid_callback_url_is_persisted_but_never_exposed_in_response(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)
        callback_url = "https://example.com/eval-done"

        resp = client.post(
            V2_EVALS,
            json={
                "experiment_name": "v2-with-callback",
                "dataset_id": dataset.id,
                "config_id": str(config.id),
                "config_version": 1,
                "callback_url": callback_url,
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 200, resp.text
        assert "callback_url" not in resp.json()["data"]
        run = db.get(EvaluationRun, resp.json()["data"]["id"])
        assert run.callback_url == callback_url

    def test_invalid_callback_url_is_rejected_and_no_run_dispatched(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ):
        """A non-HTTPS callback_url is rejected before any dispatch happens."""
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        resp = client.post(
            V2_EVALS,
            json={
                "experiment_name": "v2-bad-callback",
                "dataset_id": dataset.id,
                "config_id": str(config.id),
                "config_version": 1,
                "callback_url": "http://insecure.example.com/hook",
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 422, resp.text
        assert "invalid_callback_url" in resp.text
        _patch_dispatch.assert_not_called()


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
        assert not body["is_judge_run"]
        run = db.get(EvaluationRun, body["id"])
        assert not run.is_judge_run
        assert run.run_mode == "fast"
