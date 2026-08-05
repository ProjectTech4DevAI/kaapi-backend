from app.models.llm.request import build_kaapi_completion_config
from app.services.llm.mappers import kaapi_params_as_dict


class TestKaapiCompletionConfigTemperature:
    """Test temperature handling in KaapiCompletionConfig / kaapi_params_as_dict."""

    def test_temperature_preserved_when_user_provides_it(self) -> None:
        """When user explicitly provides temperature, it should be in params."""
        config = build_kaapi_completion_config(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
                "temperature": 0.7,
            },
        )

        assert config.params.temperature == 0.7
        assert kaapi_params_as_dict(config.params)["temperature"] == 0.7

    def test_temperature_excluded_when_user_does_not_provide_it(self) -> None:
        """When user does not provide temperature, kaapi_params_as_dict should
        strip it even though TextLLMParams has a default of 0.1."""
        config = build_kaapi_completion_config(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
            },
        )

        assert "temperature" not in kaapi_params_as_dict(config.params)

    def test_temperature_zero_preserved_when_explicitly_set(self) -> None:
        """When user explicitly sets temperature to 0.0, it should be preserved."""
        config = build_kaapi_completion_config(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
                "temperature": 0.0,
            },
        )

        assert config.params.temperature == 0.0
        assert kaapi_params_as_dict(config.params)["temperature"] == 0.0

    def test_unset_temperature_survives_json_round_trip(self) -> None:
        """A dump -> revalidate cycle (Celery request_data, persisted config
        blobs) must not bake the temperature default into the wire format,
        or the worker would forward temperature=0.1 the user never set."""
        from app.models.llm.request import ConfigBlob

        blob = ConfigBlob(
            completion=build_kaapi_completion_config(
                provider="openai",
                type="text",
                params={"model": "gpt-4o"},
            )
        )
        dumped = blob.model_dump(mode="json")
        assert "temperature" not in dumped["completion"]["params"]
        assert None not in dumped["completion"]["params"].values()

        round_tripped = ConfigBlob.model_validate(dumped)
        assert "temperature" not in kaapi_params_as_dict(
            round_tripped.completion.params
        )
