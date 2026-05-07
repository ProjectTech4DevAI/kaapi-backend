from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext


def test_evaluation_cron_job_success(
    client: TestClient,
    superuser_api_key: TestAuthContext,
) -> None:
    """Test successful cron job execution."""
    mock_result = {
        "status": "success",
        "total_processed": 5,
        "total_failed": 0,
        "total_still_processing": 1,
        "results": [
            {
                "run_id": 1,
                "run_name": "test_run",
                "action": "processed",
            }
        ],
    }

    with patch(
        "app.api.routes.cron.process_all_pending_evaluations",
        new=AsyncMock(return_value=mock_result),
    ), patch(
        "app.crud.assessment.cron.poll_all_pending_assessment_evaluations",
        new=AsyncMock(
            return_value={"processed": 0, "failed": 0, "still_processing": 0}
        ),
    ):
        response = client.get(
            f"{settings.API_V1_STR}/cron/evaluations",
            headers={"X-API-KEY": superuser_api_key.key},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_processed"] == 5
    assert data["total_failed"] == 0
    assert data["total_still_processing"] == 1


def test_evaluation_cron_job_no_pending(
    client: TestClient,
    superuser_api_key: TestAuthContext,
) -> None:
    """Test cron job when no pending evaluations exist."""
    mock_result = {
        "status": "success",
        "total_processed": 0,
        "total_failed": 0,
        "total_still_processing": 0,
        "results": [],
    }

    with patch(
        "app.api.routes.cron.process_all_pending_evaluations",
        new=AsyncMock(return_value=mock_result),
    ), patch(
        "app.crud.assessment.cron.poll_all_pending_assessment_evaluations",
        new=AsyncMock(
            return_value={"processed": 0, "failed": 0, "still_processing": 0}
        ),
    ):
        response = client.get(
            f"{settings.API_V1_STR}/cron/evaluations",
            headers={"X-API-KEY": superuser_api_key.key},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_processed"] == 0


def test_evaluation_cron_job_with_failures(
    client: TestClient,
    superuser_api_key: TestAuthContext,
) -> None:
    """Test cron job execution with some failed evaluations."""
    mock_result = {
        "status": "success",
        "total_processed": 3,
        "total_failed": 2,
        "total_still_processing": 0,
        "results": [
            {
                "run_id": 1,
                "run_name": "test_run",
                "action": "failed",
                "error": "Check failed",
            }
        ],
    }

    with patch(
        "app.api.routes.cron.process_all_pending_evaluations",
        new=AsyncMock(return_value=mock_result),
    ), patch(
        "app.crud.assessment.cron.poll_all_pending_assessment_evaluations",
        new=AsyncMock(
            return_value={"processed": 0, "failed": 0, "still_processing": 0}
        ),
    ):
        response = client.get(
            f"{settings.API_V1_STR}/cron/evaluations",
            headers={"X-API-KEY": superuser_api_key.key},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_failed"] == 2
    assert data["total_processed"] == 3


def test_evaluation_cron_job_merges_assessment_totals(
    client: TestClient,
    superuser_api_key: TestAuthContext,
) -> None:
    """Test cron job aggregates standard + assessment poll totals."""
    mock_result = {
        "status": "success",
        "total_processed": 2,
        "total_failed": 1,
        "total_still_processing": 3,
        "results": [],
    }
    assessment_result = {"processed": 4, "failed": 2, "still_processing": 1}

    with patch(
        "app.api.routes.cron.process_all_pending_evaluations",
        new=AsyncMock(return_value=mock_result),
    ), patch(
        "app.crud.assessment.cron.poll_all_pending_assessment_evaluations",
        new=AsyncMock(return_value=assessment_result),
    ):
        response = client.get(
            f"{settings.API_V1_STR}/cron/evaluations",
            headers={"X-API-KEY": superuser_api_key.key},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_processed"] == 6
    assert data["total_failed"] == 3
    assert data["total_still_processing"] == 4
    assert data["assessment"] == assessment_result


def test_evaluation_cron_job_assessment_polling_failure(
    client: TestClient,
    superuser_api_key: TestAuthContext,
) -> None:
    """Test cron keeps response successful when assessment polling fails."""
    mock_result = {
        "status": "success",
        "total_processed": 3,
        "total_failed": 0,
        "total_still_processing": 2,
        "results": [],
    }

    with patch(
        "app.api.routes.cron.process_all_pending_evaluations",
        new=AsyncMock(return_value=mock_result),
    ), patch(
        "app.crud.assessment.cron.poll_all_pending_assessment_evaluations",
        new=AsyncMock(side_effect=RuntimeError("assessment poll failure")),
    ):
        response = client.get(
            f"{settings.API_V1_STR}/cron/evaluations",
            headers={"X-API-KEY": superuser_api_key.key},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_processed"] == 3
    assert data["total_failed"] == 0
    assert data["total_still_processing"] == 2
    assert "assessment poll failure" in data["assessment_error"]


def test_evaluation_cron_job_requires_superuser(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    """Test that non-superuser cannot access cron endpoint."""
    response = client.get(
        f"{settings.API_V1_STR}/cron/evaluations",
        headers={"X-API-KEY": user_api_key.key},
    )

    assert response.status_code == 403
    response_data = response.json()
    assert "Insufficient permissions" in response_data["error"]
    assert "superuser" in response_data["error"].lower()


def test_evaluation_cron_job_not_in_schema(
    client: TestClient,
) -> None:
    """Test that cron endpoint is not included in OpenAPI schema."""
    response = client.get(f"{settings.API_V1_STR}/openapi.json")
    assert response.status_code == 200

    openapi_schema = response.json()
    paths = openapi_schema.get("paths", {})

    # Endpoint should not be in the schema due to include_in_schema=False
    assert f"{settings.API_V1_STR}/cron/evaluations" not in paths
