from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.crud.model_config import get_default_model_for_type


def test_list_models(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["count"] > 0
    assert all(m["is_active"] for m in body["data"]["data"])


def test_list_models_filter_by_provider(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/?provider=openai&limit=5",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    data = response.json()["data"]["data"]
    assert len(data) <= 5
    assert all(m["provider"] == "openai" for m in data)


def test_get_model(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/openai/gpt-4o",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    model = response.json()["data"]
    assert model["provider"] == "openai"
    assert model["model_name"] == "gpt-4o"


def test_get_model_not_found(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/openai/does-not-exist",
        headers=superuser_token_headers,
    )

    assert response.status_code == 404
    assert response.json()["error"] == "Model not found"


def test_get_default_model_for_type(db: Session) -> None:
    model = get_default_model_for_type(session=db, completion_type="text")

    assert model is not None
    assert model.default_for == "text"
    assert model.is_active is True
