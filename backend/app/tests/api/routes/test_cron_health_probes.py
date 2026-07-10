from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext


def test_health_probes_cron_enqueues_and_returns_task_id(
    client: TestClient,
    superuser_api_key: TestAuthContext,
) -> None:
    fake_async_result = SimpleNamespace(id="test-task-id")

    with patch(
        "app.api.routes.cron.run_health_probes.delay",
        return_value=fake_async_result,
    ) as delay_mock:
        response = client.get(
            f"{settings.API_V1_STR}/cron/health-probes",
            headers={"X-API-KEY": superuser_api_key.key},
        )

    assert response.status_code == 200
    assert response.json() == {"enqueued": True, "task_id": "test-task-id"}
    delay_mock.assert_called_once_with()


def test_health_probes_cron_requires_superuser(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    with patch(
        "app.api.routes.cron.run_health_probes.delay",
        return_value=SimpleNamespace(id="should-not-run"),
    ) as delay_mock:
        response = client.get(
            f"{settings.API_V1_STR}/cron/health-probes",
            headers={"X-API-KEY": user_api_key.key},
        )

    assert response.status_code == 403
    assert "Insufficient permissions" in response.json()["error"]
    delay_mock.assert_not_called()


def test_health_probes_cron_requires_authentication(
    client: TestClient,
) -> None:
    with patch(
        "app.api.routes.cron.run_health_probes.delay",
        return_value=SimpleNamespace(id="should-not-run"),
    ) as delay_mock:
        response = client.get(f"{settings.API_V1_STR}/cron/health-probes")

    assert response.status_code in (401, 403)
    delay_mock.assert_not_called()


def test_health_probes_cron_not_in_openapi_schema(client: TestClient) -> None:
    response = client.get(f"{settings.API_V1_STR}/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    assert f"{settings.API_V1_STR}/cron/health-probes" not in paths
