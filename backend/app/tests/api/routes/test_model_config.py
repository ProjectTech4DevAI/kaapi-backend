from fastapi.testclient import TestClient

from app.core.config import settings


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
    assert "has_more" in body["metadata"]
    assert body["data"]["count"] > 0
    assert all(m["is_active"] for m in body["data"]["data"])


def test_list_models_has_more(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/?skip=0&limit=1",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["count"] == 1
    assert body["metadata"]["has_more"] is True


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


def test_list_models_invalid_limit(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/?skip=0&limit=0",
        headers=superuser_token_headers,
    )
    assert response.status_code == 422


def test_get_model(client: TestClient, superuser_token_headers: dict[str, str]) -> None:
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


def test_list_models_grouped(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/grouped",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "has_more" in body["metadata"]

    grouped_models = body["data"]
    assert grouped_models
    for provider, models in grouped_models.items():
        assert isinstance(provider, str)
        assert isinstance(models, list)
        assert all(model["provider"] == provider for model in models)
        assert all(model["is_active"] for model in models)


def test_list_models_grouped_has_more(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/grouped?skip=0&limit=1",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["metadata"]["has_more"] is True


def test_list_models_grouped_invalid_limit(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/grouped?skip=0&limit=0",
        headers=superuser_token_headers,
    )

    assert response.status_code == 422


def test_list_providers(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models/providers",
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True

    providers = body["data"]
    assert isinstance(providers, list)
    assert providers == sorted(providers)
    assert len(providers) == len(set(providers))
    assert "openai" in providers
