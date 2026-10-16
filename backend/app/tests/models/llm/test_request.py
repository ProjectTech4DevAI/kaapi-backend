from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.llm.request import (
    ConfigBlob,
    InlineValidator,
    NativeCompletionConfig,
    Validator,
    build_kaapi_completion_config,
)
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


class TestTextLLMParamsThinkingLevel:
    @pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
    def test_accepts_gemini_thinking_levels(self, level: str) -> None:
        assert TextLLMParams(thinking_level=level).thinking_level == level

    def test_rejects_unknown_thinking_level(self) -> None:
        with pytest.raises(ValidationError):
            TextLLMParams(thinking_level="extreme")
def _blob(input_guardrails: list) -> ConfigBlob:
    return ConfigBlob(
        completion=NativeCompletionConfig(
            provider="openai-native", type="text", params={"model": "gpt-4o"}
        ),
        input_guardrails=input_guardrails,
    )


class TestGuardrailEntryParsing:
    def test_dict_with_config_id_parses_to_validator(self) -> None:
        config_id = uuid4()
        blob = _blob([{"validator_config_id": str(config_id)}])

        (entry,) = blob.input_guardrails or []
        assert isinstance(entry, Validator)
        assert entry.validator_config_id == config_id
        assert entry.is_enabled is True

    def test_dict_with_type_parses_to_inline_and_keeps_extras(self) -> None:
        blob = _blob(
            [
                {
                    "type": "topic_relevance_llm",
                    "on_fail": "fix",
                    "languages": ["en", "hi"],
                }
            ]
        )

        (entry,) = blob.input_guardrails or []
        assert isinstance(entry, InlineValidator)
        assert entry.model_dump() == {
            "type": "topic_relevance_llm",
            "on_fail": "fix",
            "is_enabled": True,
            "languages": ["en", "hi"],
        }

    def test_ref_dict_carrying_type_still_parses_to_validator(self) -> None:
        # InlineValidator allows extras, so without the discriminator a smart
        # union could claim this ref.
        config_id = uuid4()
        blob = _blob([{"validator_config_id": str(config_id), "type": "ban_list"}])

        (entry,) = blob.input_guardrails or []
        assert isinstance(entry, Validator)
        assert entry.model_dump() == {
            "validator_config_id": config_id,
            "is_enabled": True,
        }

    def test_unsupported_inline_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _blob([{"type": "ban_list"}])

    def test_revalidating_parsed_entries_keeps_types(self) -> None:
        parsed = _blob(
            [{"validator_config_id": str(uuid4())}, {"type": "pii_remover"}]
        ).input_guardrails

        rebuilt = _blob(list(parsed or [])).input_guardrails or []

        assert [type(e) for e in rebuilt] == [Validator, InlineValidator]

    def test_is_enabled_false_survives_json_round_trip(self) -> None:
        config_id = uuid4()
        dumped = _blob(
            [
                {"validator_config_id": str(config_id), "is_enabled": False},
                {"type": "pii_remover", "is_enabled": False},
            ]
        ).model_dump(mode="json")

        ref, inline = ConfigBlob.model_validate(dumped).input_guardrails or []

        assert isinstance(ref, Validator)
        assert ref.validator_config_id == config_id
        assert ref.is_enabled is False
        assert isinstance(inline, InlineValidator)
        assert inline.is_enabled is False
