from fastapi.testclient import TestClient

from app.core.config import settings


def test_health_reports_status_and_sha(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["sha"] == settings.GIT_SHA
