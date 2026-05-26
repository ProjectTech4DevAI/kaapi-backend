import uuid

from fastapi.testclient import TestClient

from app.core.config import settings


def _payload(model_name: str | None = None, **overrides) -> dict:
    base = {
        "provider": "google",
        "model_name": model_name or f"test-{uuid.uuid4().hex[:8]}",
        "completion_type": ["text"],
        "config": {},
        "input_modalities": ["TEXT"],
        "output_modalities": ["TEXT"],
        "pricing": None,
        "is_active": True,
    }
    base.update(overrides)
    return base


def _delete(client: TestClient, headers: dict, provider: str, model_name: str) -> None:
    client.delete(
        f"{settings.API_V1_STR}/models/{provider}/{model_name}",
        headers=headers,
    )


def test_list_models(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models",
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
        f"{settings.API_V1_STR}/models?skip=0&limit=1",
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
        f"{settings.API_V1_STR}/models?provider=openai&limit=5",
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
        f"{settings.API_V1_STR}/models?skip=0&limit=0",
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


def test_create_model_single(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(completion_type=["text", "stt"])
    response = client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )
    assert response.status_code == 201
    data = response.json()["data"]
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["model_name"] == payload["model_name"]
    assert data[0]["completion_type"] == ["text", "stt"]

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])


def test_create_model_multiple(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payloads = [_payload(), _payload()]
    response = client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payloads,
    )
    assert response.status_code == 201
    data = response.json()["data"]
    assert len(data) == 2
    names = {m["model_name"] for m in data}
    assert names == {p["model_name"] for p in payloads}

    for p in payloads:
        _delete(client, superuser_token_headers, p["provider"], p["model_name"])


def test_create_model_invalid_modality(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(input_modalities=["INVALID"])
    response = client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )
    assert response.status_code == 422


def test_create_model_invalid_completion_type(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(completion_type=["invalid"])
    response = client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )
    assert response.status_code == 422


def test_update_model_replaces_completion_type(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(completion_type=["stt"])
    client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )

    response = client.patch(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
        json={"completion_type": ["text", "stt"]},
    )
    assert response.status_code == 200
    assert set(response.json()["data"]["completion_type"]) == {"text", "stt"}

    response = client.patch(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
        json={"completion_type": ["text"]},
    )
    assert response.status_code == 200
    assert response.json()["data"]["completion_type"] == ["text"]

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])


def test_update_model_only_sent_fields_changed(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(
        completion_type=["text"],
        pricing={"response": {"input_token_cost": 1.0, "output_token_cost": 2.0}},
    )
    client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )

    response = client.patch(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
        json={"is_active": False},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["is_active"] is False
    assert data["completion_type"] == ["text"]
    assert data["pricing"]["response"]["input_token_cost"] == 1.0

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])


def test_update_model_not_found(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.patch(
        f"{settings.API_V1_STR}/models/google/does-not-exist",
        headers=superuser_token_headers,
        json={"is_active": False},
    )
    assert response.status_code == 404


def test_update_model_invalid_modality(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload()
    client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )

    response = client.patch(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
        json={"input_modalities": ["BOGUS"]},
    )
    assert response.status_code == 422

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])


def test_bulk_update_models(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    p1 = _payload(completion_type=["stt"])
    p2 = _payload(completion_type=["tts"])
    client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=[p1, p2],
    )

    response = client.patch(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=[
            {
                "provider": p1["provider"],
                "model_name": p1["model_name"],
                "completion_type": ["text", "stt"],
            },
            {
                "provider": p2["provider"],
                "model_name": p2["model_name"],
                "is_active": False,
            },
        ],
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 2
    by_name = {m["model_name"]: m for m in data}
    assert set(by_name[p1["model_name"]]["completion_type"]) == {"text", "stt"}
    assert by_name[p2["model_name"]]["is_active"] is False

    for p in [p1, p2]:
        _delete(client, superuser_token_headers, p["provider"], p["model_name"])


def test_bulk_update_atomic_on_missing_target(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(completion_type=["stt"])
    client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )

    response = client.patch(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=[
            {
                "provider": payload["provider"],
                "model_name": payload["model_name"],
                "completion_type": ["text"],
            },
            {
                "provider": "google",
                "model_name": "definitely-does-not-exist",
                "is_active": False,
            },
        ],
    )
    assert response.status_code == 404

    # First item should not have been updated either
    check = client.get(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
    )
    assert check.json()["data"]["completion_type"] == ["stt"]

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])


def test_delete_model(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload()
    client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )

    response = client.delete(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200

    check = client.get(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
    )
    assert check.status_code == 404


def test_delete_model_not_found(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.delete(
        f"{settings.API_V1_STR}/models/google/does-not-exist",
        headers=superuser_token_headers,
    )
    assert response.status_code == 404


def test_create_model_duplicate_fails(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload()
    r1 = client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )
    assert r1.status_code == 201

    r2 = client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )
    assert r2.status_code >= 400

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])


def test_create_invalid_provider(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(provider="nonexistent_provider")
    response = client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )
    assert response.status_code == 422


def test_update_empty_body_noop(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(completion_type=["stt"])
    client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )

    response = client.patch(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
        json={},
    )
    assert response.status_code == 200
    assert response.json()["data"]["completion_type"] == ["stt"]

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])


def test_bulk_update_empty_array(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.patch(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=[],
    )
    assert response.status_code == 200
    assert response.json()["data"] == []


def test_list_models_invalid_provider_filter(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/models?provider=bogus",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    assert response.json()["data"]["count"] == 0


def test_inactive_model_excluded_from_list(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(is_active=False)
    client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )

    response = client.get(
        f"{settings.API_V1_STR}/models?provider={payload['provider']}",
        headers=superuser_token_headers,
    )
    names = [m["model_name"] for m in response.json()["data"]["data"]]
    assert payload["model_name"] not in names

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])


def test_update_pricing_replaces_full_object(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    payload = _payload(
        pricing={
            "response": {"input_token_cost": 1.0, "output_token_cost": 2.0},
            "batch": {"input_token_cost": 0.5, "output_token_cost": 1.0},
        },
    )
    client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )

    response = client.patch(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
        json={
            "pricing": {
                "response": {"input_token_cost": 5.0, "output_token_cost": 10.0}
            }
        },
    )
    assert response.status_code == 200
    pricing = response.json()["data"]["pricing"]
    assert pricing["response"]["input_token_cost"] == 5.0
    assert "batch" not in pricing  # full replace, not merge

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])


def test_create_with_multiple_completion_types_and_query(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    """Model supporting both text + stt should appear when filtering by either type via validation."""
    payload = _payload(
        completion_type=["text", "stt"],
        input_modalities=["TEXT", "AUDIO"],
    )
    response = client.post(
        f"{settings.API_V1_STR}/models",
        headers=superuser_token_headers,
        json=payload,
    )
    assert response.status_code == 201
    created = response.json()["data"][0]
    assert set(created["completion_type"]) == {"text", "stt"}

    fetched = client.get(
        f"{settings.API_V1_STR}/models/{payload['provider']}/{payload['model_name']}",
        headers=superuser_token_headers,
    )
    assert set(fetched.json()["data"]["completion_type"]) == {"text", "stt"}

    _delete(client, superuser_token_headers, payload["provider"], payload["model_name"])
