from typing import Any
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext


def _fake_tick_result(**overrides: Any) -> dict[str, Any]:
    result = {
        "enqueued": True,
        "job_id": str(uuid4()),
        "probe_index": 4,
        "previous_job_status": "SUCCESS",
    }
    result.update(overrides)
    return result


def test_health_probes_cron_runs_tick_and_returns_result(
    client: TestClient,
    superuser_api_key: TestAuthContext,
) -> None:
    canned = _fake_tick_result()

    with patch(
        "app.api.routes.cron.run_health_probe_tick", return_value=canned
    ) as tick_mock:
        response = client.get(
            f"{settings.API_V1_STR}/cron/health-probes",
            headers={"X-API-KEY": superuser_api_key.key},
        )

    assert response.status_code == 200
    assert response.json() == canned
    tick_mock.assert_called_once_with()


def test_health_probes_cron_requires_superuser(
    client: TestClient,
    user_api_key: TestAuthContext,
) -> None:
    with patch(
        "app.api.routes.cron.run_health_probe_tick",
        return_value=_fake_tick_result(),
    ) as tick_mock:
        response = client.get(
            f"{settings.API_V1_STR}/cron/health-probes",
            headers={"X-API-KEY": user_api_key.key},
        )

    assert response.status_code == 403
    assert "Insufficient permissions" in response.json()["error"]
    tick_mock.assert_not_called()


def test_health_probes_cron_requires_authentication(client: TestClient) -> None:
    with patch(
        "app.api.routes.cron.run_health_probe_tick",
        return_value=_fake_tick_result(),
    ) as tick_mock:
        response = client.get(f"{settings.API_V1_STR}/cron/health-probes")

    assert response.status_code in (401, 403)
    tick_mock.assert_not_called()


def test_health_probes_cron_not_in_openapi_schema(client: TestClient) -> None:
    response = client.get(f"{settings.API_V1_STR}/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    assert f"{settings.API_V1_STR}/cron/health-probes" not in paths
