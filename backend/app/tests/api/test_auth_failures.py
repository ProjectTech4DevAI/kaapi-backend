import pytest
from fastapi.testclient import TestClient

from app.core.config import settings


PROTECTED_ENDPOINTS = [
    (f"{settings.API_V1_STR}/collections/", "GET"),
    (f"{settings.API_V1_STR}/collections/", "POST"),
    (f"{settings.API_V1_STR}/collections/12345678-1234-5678-1234-567812345678", "GET"),
    (
        f"{settings.API_V1_STR}/collections/12345678-1234-5678-1234-567812345678",
        "DELETE",
    ),
    (
        f"{settings.API_V1_STR}/collections/jobs/12345678-1234-5678-1234-567812345678",
        "GET",
    ),
    (f"{settings.API_V1_STR}/documents/", "GET"),
    (f"{settings.API_V1_STR}/documents/", "POST"),
    (f"{settings.API_V1_STR}/documents/12345678-1234-5678-1234-567812345678", "GET"),
    (f"{settings.API_V1_STR}/documents/12345678-1234-5678-1234-567812345678", "DELETE"),
    (
        f"{settings.API_V1_STR}/documents/transformation/12345678-1234-5678-1234-567812345678",
        "GET",
    ),
    (f"{settings.API_V1_STR}/cron/evaluations", "GET"),
    (f"{settings.API_V1_STR}/evaluations/datasets/", "POST"),
    (f"{settings.API_V1_STR}/evaluations/datasets/", "GET"),
    (
        f"{settings.API_V1_STR}/evaluations/datasets/12345678-1234-5678-1234-567812345678",
        "GET",
    ),
    (f"{settings.API_V1_STR}/evaluations", "POST"),
    (f"{settings.API_V1_STR}/evaluations", "GET"),
    (f"{settings.API_V1_STR}/evaluations/12345678-1234-5678-1234-567812345678", "GET"),
    (f"{settings.API_V1_STR}/llm/call", "POST"),
]


@pytest.mark.parametrize("endpoint,method", PROTECTED_ENDPOINTS)
def test_endpoints_reject_missing_auth_header(
    client: TestClient, endpoint: str, method: str
) -> None:
    """Test that all protected endpoints return 401 when no auth header is provided."""
    kwargs = {"json": {"name": "test"}} if method in ["POST", "PATCH"] else {}
    response = client.request(method, endpoint, **kwargs)

    assert (
        response.status_code == 401
    ), f"Expected 401 for {method} {endpoint} without auth, got {response.status_code}"


@pytest.mark.parametrize("endpoint,method", PROTECTED_ENDPOINTS)
def test_endpoints_reject_invalid_auth_format(
    client: TestClient, endpoint: str, method: str
) -> None:
    """Test that all protected endpoints return 401 when auth header has invalid format."""
    kwargs = {"json": {"name": "test"}} if method in ["POST", "PATCH"] else {}
    response = client.request(
        method, endpoint, headers={"Authorization": "InvalidFormat"}, **kwargs
    )

    assert (
        response.status_code == 401
    ), f"Expected 401 for {method} {endpoint} with invalid format, got {response.status_code}"


@pytest.mark.parametrize("endpoint,method", PROTECTED_ENDPOINTS)
def test_endpoints_reject_nonexistent_api_key(
    client: TestClient, endpoint: str, method: str
) -> None:
    """Test that all protected endpoints return 401 when API key doesn't exist."""
    kwargs = {"json": {"name": "test"}} if method in ["POST", "PATCH"] else {}
    response = client.request(
        method,
        endpoint,
        headers={"Authorization": "ApiKey FakeKeyThatDoesNotExist123456789"},
        **kwargs,
    )

    assert (
        response.status_code == 401
    ), f"Expected 401 for {method} {endpoint} with fake key, got {response.status_code}"
