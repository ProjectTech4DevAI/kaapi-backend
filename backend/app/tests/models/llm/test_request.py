from app.models.llm.request import KaapiCompletionConfig


class TestKaapiCompletionConfigTemperature:
    """Test temperature handling in KaapiCompletionConfig.validate_params."""

    def test_temperature_preserved_when_user_provides_it(self) -> None:
        """When user explicitly provides temperature, it should be in params."""
        config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
                "temperature": 0.7,
            },
        )

        assert "temperature" in config.params
        assert config.params["temperature"] == 0.7

    def test_temperature_excluded_when_user_does_not_provide_it(self) -> None:
        """When user does not provide temperature, it should NOT be in params
        even though TextLLMParams has a default of 0.1."""
        config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
            },
        )

        assert "temperature" not in config.params

    def test_temperature_zero_preserved_when_explicitly_set(self) -> None:
        """When user explicitly sets temperature to 0.0, it should be preserved."""
        config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
                "temperature": 0.0,
            },
        )

        assert "temperature" in config.params
        assert config.params["temperature"] == 0.0


# Model-allowlist enforcement moved from KaapiCompletionConfig.validate_params to
# the CRUD layer (crud.model_config.validate_blob_model_or_raise) which consults
# the model_config table. See tests/crud/config/* for coverage.
