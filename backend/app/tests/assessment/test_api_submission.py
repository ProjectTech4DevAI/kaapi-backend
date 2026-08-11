"""Tests for the BATCH API-client submit entrypoint (app/services/assessment/api/submission.py)."""

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlmodel import select

from app.core.config import settings
from app.crud.assessment import api
from app.models.assessment import (
    Assessment,
    AssessmentCreate,
    AssessmentStatus,
)
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.services.assessment.api import submission
from app.tests.utils.auth import get_user_test_auth_context
from app.tests.utils.test_data import create_test_config
from app.tests.utils.utils import random_lower_string


def _assessment_config(
    db, project_id, *, provider="openai", model="gpt-4o", input_schema=None
):
    """ASSESSMENT config with the mandatory typed ``input_schema`` (default one text
    column ``a``); callers pass ``input_schema`` to exercise row-validation branches."""
    params = {
        "model": model,
        "json_output_schema": {
            "type": "object",
            "properties": {"s": {"type": "integer"}},
        },
        "input_schema": input_schema or {"a": {"type": "text"}},
    }
    blob = AssessmentConfigBlob.model_validate(
        {"assessment": {"provider": provider, "type": "text", "params": params}}
    )
    return create_test_config(
        db,
        project_id=project_id,
        name=f"assess-{random_lower_string()}",
        config_blob=blob,
        tag=ConfigTag.ASSESSMENT,
    )


def _request(config, data):
    return AssessmentCreate.model_validate(
        {
            "config": {"id": str(config.id), "version": 1},
            "input": {"query": "assess {a}", "data": data},
            "callback_url": "https://hook.example/cb",
            "request_metadata": {"ref": "abc"},
        }
    )


class TestSubmit:
    @pytest.fixture(autouse=True)
    def _bypass_callback_check(self):
        # These cases exercise config resolution, row validation and dispatch — not the
        # SSRF callback guard. Bypass its DNS-backed check so they don't depend on live
        # name resolution; the guard itself is covered by TestCallbackUrlValidation.
        with patch("app.services.assessment.api.submission.validate_callback_url"):
            yield

    def test_creates_assessment_run_and_dispatches(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(db, auth.project_id)
        request = _request(config, [{"a": "one"}, {"a": "two"}])

        with patch("app.celery.tasks.job_execution.run_assessment_api_batch") as task:
            response = submission.submit(
                session=db,
                request=request,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )

        assert response.status == AssessmentStatus.PROCESSING
        assert response.message == "Your assessment is being processed"
        task.delay.assert_called_once()

        # Parent + execution persisted with the seeded runtime bag and PROCESSING status.
        assessment = db.get(Assessment, response.assessment_id)
        assert assessment.status == AssessmentStatus.PROCESSING
        executions = api.list_executions(
            session=db, assessment_id=response.assessment_id
        )
        assert len(executions) == 1
        execution = executions[0]
        assert execution.total_items == 2
        bag = execution.execution
        assert bag["provider"] == "openai"
        assert bag["gate_passed"] == [True, True]
        assert bag["stage"] == "assessment"
        assert bag["stage_status"] == AssessmentStatus.PENDING.value
        assert bag["callback_url"].startswith("https://hook.example")

    def test_row_missing_declared_column_is_422(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(
            db,
            auth.project_id,
            input_schema={"a": {"type": "text"}, "b": {"type": "text"}},
        )
        request = _request(config, [{"a": "present"}, {"a": "present", "b": "here"}])

        with pytest.raises(HTTPException) as exc:
            submission.submit(
                session=db,
                request=request,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 422
        assert "input.data[0]" in exc.value.detail
        assert "missing required column" in exc.value.detail

    def test_row_undeclared_column_is_422(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(
            db, auth.project_id, input_schema={"a": {"type": "text"}}
        )
        request = _request(config, [{"a": "one"}, {"a": "two", "extra": "nope"}])

        with pytest.raises(HTTPException) as exc:
            submission.submit(
                session=db,
                request=request,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 422
        assert "input.data[1]" in exc.value.detail
        assert "not declared in input_schema" in exc.value.detail

    def test_row_attachment_value_not_url_is_422(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(
            db,
            auth.project_id,
            input_schema={"img": {"type": "image", "format": "url"}},
        )
        request = _request(config, [{"img": "not-a-url"}])

        with pytest.raises(HTTPException) as exc:
            submission.submit(
                session=db,
                request=request,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 422
        assert "input.data[0]" in exc.value.detail
        assert "must be a URL" in exc.value.detail

    def test_unsupported_provider_is_422(self, db) -> None:
        auth = get_user_test_auth_context(db)
        # Build a config whose stored blob names an unsupported batch provider.
        config = _assessment_config(db, auth.project_id, provider="openai")
        request = _request(config, [{"a": "one"}])
        with patch(
            "app.services.assessment.api.submission.batch_service.is_supported_provider",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc:
                submission.submit(
                    session=db,
                    request=request,
                    organization_id=auth.organization_id,
                    project_id=auth.project_id,
                )
        assert exc.value.status_code == 422
        assert "not supported" in exc.value.detail

    def test_config_not_found_is_404(self, db) -> None:
        auth = get_user_test_auth_context(db)
        request = AssessmentCreate.model_validate(
            {
                "config": {"id": str(uuid4()), "version": 1},
                "input": {"query": "assess {a}", "data": [{"a": "1"}]},
                "callback_url": "https://hook.example/cb",
            }
        )
        with pytest.raises(HTTPException) as exc:
            submission.submit(
                session=db,
                request=request,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 404

    def test_response_input_method_is_501(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(db, auth.project_id)
        request = AssessmentCreate.model_validate(
            {
                "config": {"id": str(config.id), "version": 1},
                "input": {"query": "single query"},
                "callback_url": "https://hook.example/cb",
            }
        )
        with pytest.raises(HTTPException) as exc:
            submission.submit(
                session=db,
                request=request,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 501

    def test_base64_attachment_column_is_422(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(
            db,
            auth.project_id,
            input_schema={"img": {"type": "image", "format": "base64"}},
        )
        # URL value clears row-validation so build_rows is what rejects base64-format.
        request = _request(config, [{"img": "https://x/a.png"}])
        with pytest.raises(HTTPException) as exc:
            submission.submit(
                session=db,
                request=request,
                organization_id=auth.organization_id,
                project_id=auth.project_id,
            )
        assert exc.value.status_code == 422
        assert "url-format" in exc.value.detail

    def test_dispatch_failure_marks_failed_and_503(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(db, auth.project_id)
        request = _request(config, [{"a": "one"}])

        with patch("app.celery.tasks.job_execution.run_assessment_api_batch") as task:
            task.delay.side_effect = RuntimeError("broker down")
            with pytest.raises(HTTPException) as exc:
                submission.submit(
                    session=db,
                    request=request,
                    organization_id=auth.organization_id,
                    project_id=auth.project_id,
                )
        assert exc.value.status_code == 503

        # The just-created assessment was flipped to FAILED before the 503 surfaced.
        latest = db.exec(
            select(Assessment)
            .where(Assessment.project_id == auth.project_id)
            .order_by(Assessment.inserted_at.desc())
        ).first()
        assert latest.status == AssessmentStatus.FAILED


class TestCallbackUrlValidation:
    """BUG 3 regression: submission.submit validates callback_url up front (HTTPS +
    SSRF/private-IP guard) and maps failure to 422, instead of only at delivery time."""

    def _submit(self, db, auth, config, callback_url):
        request = AssessmentCreate.model_validate(
            {
                "config": {"id": str(config.id), "version": 1},
                "input": {"query": "assess {a}", "data": [{"a": "one"}]},
                "callback_url": callback_url,
            }
        )
        return submission.submit(
            session=db,
            request=request,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )

    def test_non_https_callback_is_422(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(db, auth.project_id)
        with pytest.raises(HTTPException) as exc:
            self._submit(db, auth, config, "http://example.com/hook")
        assert exc.value.status_code == 422
        assert "HTTPS" in exc.value.detail

    def test_private_ip_callback_is_422(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(db, auth.project_id)
        with pytest.raises(HTTPException) as exc:
            self._submit(db, auth, config, "https://10.0.0.1/hook")
        assert exc.value.status_code == 422

    def test_loopback_callback_is_422(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(db, auth.project_id)
        with pytest.raises(HTTPException) as exc:
            self._submit(db, auth, config, "https://127.0.0.1/hook")
        assert exc.value.status_code == 422

    def test_valid_public_https_callback_succeeds(self, db) -> None:
        auth = get_user_test_auth_context(db)
        config = _assessment_config(db, auth.project_id)
        # Assert the guard is invoked with the exact URL rather than trusting live DNS.
        with (
            patch(
                "app.services.assessment.api.submission.validate_callback_url"
            ) as validate,
            patch("app.celery.tasks.job_execution.run_assessment_api_batch") as task,
        ):
            response = self._submit(db, auth, config, "https://example.com/hook")
        validate.assert_called_once_with("https://example.com/hook")
        assert response.status == AssessmentStatus.PROCESSING
        task.delay.assert_called_once()


class TestCreateAssessmentRoute:
    @pytest.fixture(autouse=True)
    def _bypass_callback_check(self):
        with patch("app.services.assessment.api.submission.validate_callback_url"):
            yield

    def test_batch_input_dispatches_and_returns_202_body(
        self, client: TestClient, superuser_api_key_header, superuser_api_key, db
    ) -> None:
        config = _assessment_config(db, superuser_api_key.project_id)
        body = {
            "config": {"id": str(config.id), "version": 1},
            "input": {"query": "assess {a}", "data": [{"a": "one"}]},
            "callback_url": "https://hook.example/cb",
        }
        with patch("app.celery.tasks.job_execution.run_assessment_api_batch") as task:
            resp = client.post(
                f"{settings.API_V1_STR}/assessments",
                headers=superuser_api_key_header,
                json=body,
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "PROCESSING"
        assert data["message"] == "Your assessment is being processed"
        task.delay.assert_called_once()

    def test_response_input_is_501(
        self, client: TestClient, superuser_api_key_header
    ) -> None:
        body = {
            "config": {"id": "00000000-0000-0000-0000-000000000001", "version": 1},
            "input": {"query": "single query"},
            "callback_url": "https://hook.example/cb",
        }
        resp = client.post(
            f"{settings.API_V1_STR}/assessments",
            headers=superuser_api_key_header,
            json=body,
        )
        assert resp.status_code == 501
