from fastapi.testclient import TestClient

from app.core.config import settings


# List of all protected endpoints to test
PROTECTED_ENDPOINTS = [
    # Collections
    (f"{settings.API_V1_STR}/collections/", "GET"),
    (f"{settings.API_V1_STR}/collections/", "POST"),
    (f"{settings.API_V1_STR}/collections/12345678-1234-5678-1234-567812345678", "GET"),
    (
        f"{settings.API_V1_STR}/collections/12345678-1234-5678-1234-567812345678",
        "DELETE",
    ),
    # Documents
    (f"{settings.API_V1_STR}/documents/", "GET"),
    (f"{settings.API_V1_STR}/documents/", "POST"),
    (f"{settings.API_V1_STR}/documents/12345678-1234-5678-1234-567812345678", "GET"),
    (f"{settings.API_V1_STR}/documents/12345678-1234-5678-1234-567812345678", "DELETE"),
    # Projects
    (f"{settings.API_V1_STR}/projects/", "GET"),
    (f"{settings.API_V1_STR}/projects/", "POST"),
    # Organizations
    (f"{settings.API_V1_STR}/organizations/", "GET"),
    (f"{settings.API_V1_STR}/organizations/", "POST"),
]


def test_all_endpoints_reject_missing_auth_header(client: TestClient):
    """Test that all protected endpoints return 401 when no auth header is provided."""
    for endpoint, method in PROTECTED_ENDPOINTS:
        if method == "GET":
            response = client.get(endpoint)
        elif method == "POST":
            response = client.post(endpoint, json={"name": "test"})
        elif method == "DELETE":
            response = client.delete(endpoint)
        elif method == "PUT":
            response = client.put(endpoint, json={"name": "test"})
        elif method == "PATCH":
            response = client.patch(endpoint, json={"name": "test"})

        assert (
            response.status_code == 401
        ), f"Expected 401 for {method} {endpoint} without auth, got {response.status_code}"


def test_all_endpoints_reject_invalid_auth_format(client: TestClient):
    """Test that all protected endpoints return 401 when auth header has invalid format."""
    invalid_headers = {"Authorization": "InvalidFormat"}

    for endpoint, method in PROTECTED_ENDPOINTS:
        if method == "GET":
            response = client.get(endpoint, headers=invalid_headers)
        elif method == "POST":
            response = client.post(
                endpoint, json={"name": "test"}, headers=invalid_headers
            )
        elif method == "DELETE":
            response = client.delete(endpoint, headers=invalid_headers)
        elif method == "PUT":
            response = client.put(
                endpoint, json={"name": "test"}, headers=invalid_headers
            )
        elif method == "PATCH":
            response = client.patch(
                endpoint, json={"name": "test"}, headers=invalid_headers
            )

        assert (
            response.status_code == 401
        ), f"Expected 401 for {method} {endpoint} with invalid format, got {response.status_code}"


def test_all_endpoints_reject_nonexistent_api_key(client: TestClient):
    """Test that all protected endpoints return 401 when API key doesn't exist."""
    fake_headers = {"Authorization": "ApiKey FakeKeyThatDoesNotExist123456789"}

    for endpoint, method in PROTECTED_ENDPOINTS:
        if method == "GET":
            response = client.get(endpoint, headers=fake_headers)
        elif method == "POST":
            response = client.post(
                endpoint, json={"name": "test"}, headers=fake_headers
            )
        elif method == "DELETE":
            response = client.delete(endpoint, headers=fake_headers)
        elif method == "PUT":
            response = client.put(endpoint, json={"name": "test"}, headers=fake_headers)
        elif method == "PATCH":
            response = client.patch(
                endpoint, json={"name": "test"}, headers=fake_headers
            )

        assert (
            response.status_code == 401
        ), f"Expected 401 for {method} {endpoint} with fake key, got {response.status_code}"
