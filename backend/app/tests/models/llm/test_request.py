import pytest
from pydantic import ValidationError

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

    def test_temperature_with_instructions(self) -> None:
        """Temperature should be preserved alongside other params when provided."""
        config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
                "instructions": "Be helpful",
                "temperature": 1.5,
            },
        )

        assert config.params["temperature"] == 1.5
        assert config.params["instructions"] == "Be helpful"

    def test_no_temperature_with_other_params(self) -> None:
        """When temperature is not provided, other params should still be present."""
        config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
                "instructions": "Be helpful",
                "reasoning": "high",
            },
        )

        assert "temperature" not in config.params
        assert config.params["instructions"] == "Be helpful"
        assert config.params["reasoning"] == "high"


class TestNewSupportedModels:
    """Test that newly added models are accepted for openai/text provider."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
    )
    def test_new_model_accepted(self, model: str) -> None:
        """New models should be accepted for openai text provider."""
        config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={"model": model},
        )

        assert config.params["model"] == model

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-5.4",
            "gpt-5.1",
            "gpt-5-mini",
            "gpt-5-nano",
            "o1",
            "o1-preview",
            "o1-mini",
        ],
    )
    def test_existing_models_still_accepted(self, model: str) -> None:
        """Previously supported models should still be accepted."""
        config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={"model": model},
        )

        assert config.params["model"] == model

    def test_unsupported_model_rejected(self) -> None:
        """An unsupported model should raise a validation error."""
        with pytest.raises(ValidationError, match="not supported"):
            KaapiCompletionConfig(
                provider="openai",
                type="text",
                params={"model": "unsupported-model-xyz"},
            )
