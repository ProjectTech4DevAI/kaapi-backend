from unittest.mock import patch

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
        "organizations_processed": 2,
        "total_processed": 5,
        "total_failed": 0,
        "total_still_processing": 1,
        "results": [
            {
                "org_id": superuser_api_key.organization_id,
                "org_name": "Test Org",
                "summary": {"processed": 3, "failed": 0},
            }
        ],
    }

    with patch(
        "app.api.routes.cron.process_all_pending_evaluations_sync",
        return_value=mock_result,
    ):
        response = client.get(
            f"{settings.API_V1_STR}/cron/evaluations",
            headers={"X-API-KEY": superuser_api_key.key},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["organizations_processed"] == 2
    assert data["total_processed"] == 5
    assert data["total_failed"] == 0
    assert data["total_still_processing"] == 1


def test_evaluation_cron_job_no_organizations(
    client: TestClient,
    superuser_api_key: TestAuthContext,
) -> None:
    """Test cron job when no organizations exist."""
    mock_result = {
        "status": "success",
        "organizations_processed": 0,
        "total_processed": 0,
        "total_failed": 0,
        "total_still_processing": 0,
        "message": "No organizations to process",
        "results": [],
    }

    with patch(
        "app.api.routes.cron.process_all_pending_evaluations_sync",
        return_value=mock_result,
    ):
        response = client.get(
            f"{settings.API_V1_STR}/cron/evaluations",
            headers={"X-API-KEY": superuser_api_key.key},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["organizations_processed"] == 0
    assert data["message"] == "No organizations to process"


def test_evaluation_cron_job_with_failures(
    client: TestClient,
    superuser_api_key: TestAuthContext,
) -> None:
    """Test cron job execution with some failed evaluations."""
    mock_result = {
        "status": "success",
        "organizations_processed": 1,
        "total_processed": 3,
        "total_failed": 2,
        "total_still_processing": 0,
        "results": [
            {
                "org_id": superuser_api_key.organization_id,
                "org_name": "Test Org",
                "summary": {"processed": 3, "failed": 2},
            }
        ],
    }

    with patch(
        "app.api.routes.cron.process_all_pending_evaluations_sync",
        return_value=mock_result,
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
