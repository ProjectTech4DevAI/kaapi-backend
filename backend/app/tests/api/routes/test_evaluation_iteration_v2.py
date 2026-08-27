"""Tests for `POST /api/v2/evaluations/iterations` — the eval-iterate-improve
loop trigger. The Celery enqueue (`start_evaluation_iteration_round`) and the
SSRF-checking `validate_callback_url` (real DNS resolution) are the mocked
boundaries; the DB is real.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Config, EvaluationDataset
from app.models.evaluation_iteration import EvaluationIterationRun
from app.models.llm.request import ConfigBlob, build_kaapi_completion_config
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)

ITERATIONS_URL = f"{settings.API_V2_STR}/evaluations/iterations"
_ROUTE_VALIDATE = "app.api.routes.evaluations.iteration_v2.validate_callback_url"


def _make_dataset(*, db: Session, user_api_key: TestAuthContext) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


def _make_text_config(db: Session, project_id: int) -> Config:
    blob = ConfigBlob(
        completion=build_kaapi_completion_config(
            provider="openai",
            type="text",
            params={"model": "gpt-4o-iter-route-test", "temperature": 0.7},
        )
    )
    return create_test_config(
        db=db, project_id=project_id, use_kaapi_schema=True, config_blob=blob
    )


@pytest.fixture
def _patch_dispatch():
    """Stub the Celery enqueue and skip the real-DNS SSRF check for example.com."""
    with (
        patch(_ROUTE_VALIDATE),
        patch(
            "app.services.evaluations.iteration.start_evaluation_iteration_round",
            return_value="fake-task-id",
        ) as mock_start,
    ):
        yield mock_start


class TestCreateEvaluationIterationRoute:
    def test_valid_request_returns_202_and_expected_shape(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ) -> None:
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        resp = client.post(
            ITERATIONS_URL,
            json={
                "dataset_id": dataset.id,
                "experiment_name": "route-iter",
                "config_id": str(config.id),
                "config_version": 1,
                "callback_url": "https://example.com/callback",
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 202, resp.text
        body = resp.json()["data"]
        assert body["status"] == "processing"
        assert "iteration_run_id" in body
        assert "inserted_at" in body and "updated_at" in body

        run = db.get(EvaluationIterationRun, body["iteration_run_id"])
        assert run is not None
        assert run.experiment_name == "route-iter"
        assert run.dataset_id == dataset.id
        assert run.config_id == config.id
        _patch_dispatch.assert_called_once()

    def test_invalid_callback_url_returns_422(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ) -> None:
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        resp = client.post(
            ITERATIONS_URL,
            json={
                "dataset_id": dataset.id,
                "experiment_name": "route-bad-cb",
                "config_id": str(config.id),
                "config_version": 1,
                "callback_url": "not-a-valid-url",
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 422
        _patch_dispatch.assert_not_called()

    def test_max_rounds_above_hard_cap_is_rejected(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        db: Session,
        user_api_key: TestAuthContext,
        _patch_dispatch,
    ) -> None:
        dataset = _make_dataset(db=db, user_api_key=user_api_key)
        config = _make_text_config(db, user_api_key.project_id)

        resp = client.post(
            ITERATIONS_URL,
            json={
                "dataset_id": dataset.id,
                "experiment_name": "route-clamp",
                "config_id": str(config.id),
                "config_version": 1,
                "max_rounds": settings.EVAL_ITERATION_MAX_ROUNDS_HARD_CAP + 100,
                "callback_url": "https://example.com/callback",
            },
            headers=user_api_key_header,
        )

        assert resp.status_code == 422, resp.text
        _patch_dispatch.assert_not_called()
