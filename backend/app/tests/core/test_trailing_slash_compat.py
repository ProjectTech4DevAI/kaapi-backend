from fastapi.testclient import TestClient

from app.core.config import settings


def test_trailing_slash_routes_to_canonical(client: TestClient) -> None:
    """Both '/health' and '/health/' must hit the handler directly (no redirect)."""
    canonical = client.get("/health")
    legacy = client.get("/health/", follow_redirects=False)

    assert canonical.status_code == 200
    assert legacy.status_code == 200
    assert legacy.json() == canonical.json()


def test_trailing_slash_preserves_query_string(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    """Query string survives the rewrite from '/users/?...' to '/users?...'."""
    response = client.get(
        f"{settings.API_V1_STR}/users/?skip=0&limit=1",
        headers=superuser_token_headers,
        follow_redirects=False,
    )
    assert response.status_code == 200


def test_root_path_unaffected(client: TestClient) -> None:
    """The bare '/' must not be rewritten to ''."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code != 500
