from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.config import settings


def _flag_dict(enabled: bool = True) -> dict:
    return {
        "id": 1,
        "key": "ASSESSMENT",
        "organization_id": 1,
        "project_id": 1,
        "enabled": enabled,
        "inserted_at": datetime(2024, 1, 1).isoformat(),
        "updated_at": datetime(2024, 1, 1).isoformat(),
    }


def test_create_feature_flag_success(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = {
        "key": "ASSESSMENT",
        "organization_id": 1,
        "project_id": 1,
        "enabled": True,
    }
    with patch(
        "app.api.routes.features.validate_project",
        return_value=SimpleNamespace(organization_id=1),
    ), patch(
        "app.api.routes.features.create_feature_flag",
        return_value=_flag_dict(True),
    ):
        response = client.post(
            f"{settings.API_V1_STR}/feature-flags",
            headers=superuser_token_headers,
            json=payload,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["key"] == "ASSESSMENT"
    assert body["data"]["enabled"] is True


def test_create_feature_flag_conflict(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = {
        "key": "ASSESSMENT",
        "organization_id": 1,
        "project_id": 1,
        "enabled": True,
    }
    with patch(
        "app.api.routes.features.validate_project",
        return_value=SimpleNamespace(organization_id=1),
    ), patch(
        "app.api.routes.features.create_feature_flag",
        side_effect=HTTPException(
            status_code=409,
            detail="Feature flag already exists",
        ),
    ):
        response = client.post(
            f"{settings.API_V1_STR}/feature-flags",
            headers=superuser_token_headers,
            json=payload,
        )
    assert response.status_code == 409


def test_list_feature_flags_validation_and_success(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    with patch(
        "app.api.routes.features.validate_project",
        return_value=SimpleNamespace(organization_id=1),
    ), patch(
        "app.api.routes.features.list_feature_flags",
        return_value=[_flag_dict(True)],
    ):
        ok = client.get(
            f"{settings.API_V1_STR}/feature-flags?key=ASSESSMENT&project_id=1",
            headers=superuser_token_headers,
        )
    assert ok.status_code == 200
    body = ok.json()
    assert body["success"] is True
    assert len(body["data"]) == 1
    assert body["data"][0]["key"] == "ASSESSMENT"


def test_patch_and_delete_feature_flag(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = {
        "key": "ASSESSMENT",
        "organization_id": 1,
        "project_id": 1,
        "enabled": False,
    }
    with patch(
        "app.api.routes.features.validate_project",
        return_value=SimpleNamespace(organization_id=1),
    ), patch(
        "app.api.routes.features.update_feature_flag",
        return_value=_flag_dict(False),
    ):
        patch_resp = client.patch(
            f"{settings.API_V1_STR}/feature-flags",
            headers=superuser_token_headers,
            json=payload,
        )
    assert patch_resp.status_code == 200
    patch_body = patch_resp.json()
    assert patch_body["success"] is True
    assert patch_body["data"]["enabled"] is False

    with patch(
        "app.api.routes.features.validate_project",
        return_value=SimpleNamespace(organization_id=1),
    ), patch("app.api.routes.features.delete_feature_flag"):
        delete_resp = client.request(
            "DELETE",
            f"{settings.API_V1_STR}/feature-flags",
            headers=superuser_token_headers,
            json={k: payload[k] for k in ("key", "organization_id", "project_id")},
        )
    assert delete_resp.status_code == 200
    body = delete_resp.json()
    assert body["success"] is True
    assert body["data"]["deleted"] is True
