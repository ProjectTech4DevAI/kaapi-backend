"""
Unit tests for LLM parameter mapping functions.

Tests the transformation of Kaapi-abstracted parameters to provider-native formats.
"""

import pytest

from app.models.llm.request import (
    TextLLMParams,
    STTLLMParams,
    TTSLLMParams,
    KaapiCompletionConfig,
    NativeCompletionConfig,
)
from app.services.llm.mappers import (
    map_kaapi_to_openai_params,
    map_kaapi_to_google_params,
    map_kaapi_to_sarvam_params,
    transform_kaapi_config_to_native,
)


class TestMapKaapiToOpenAIParams:
    """Test cases for map_kaapi_to_openai_params function."""

    def test_basic_model_mapping(self):
        """Test basic model parameter mapping."""
        kaapi_params = TextLLMParams(model="gpt-4o")

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        # TextLLMParams has default temperature=0.1
        assert result == {"model": "gpt-4o", "temperature": 0.1}
        assert warnings == []

    def test_instructions_mapping(self):
        """Test instructions parameter mapping."""
        kaapi_params = TextLLMParams(
            model="gpt-4",
            instructions="You are a helpful assistant.",
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-4"
        assert result["instructions"] == "You are a helpful assistant."
        assert warnings == []

    def test_temperature_mapping(self):
        """Test temperature parameter mapping for non-reasoning models."""
        kaapi_params = TextLLMParams(
            model="gpt-4",
            temperature=0.7,
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert warnings == []

    def test_temperature_zero_mapping(self):
        """Test that temperature=0 is correctly mapped (edge case)."""
        kaapi_params = TextLLMParams(
            model="gpt-4",
            temperature=0.0,
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["temperature"] == 0.0
        assert warnings == []

    def test_reasoning_mapping_for_reasoning_models(self):
        """Test reasoning parameter mapping to OpenAI format for reasoning-capable models."""
        kaapi_params = TextLLMParams(
            model="o1",
            reasoning="high",
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "o1"
        assert result["reasoning"] == {"effort": "high"}
        # Temperature is suppressed for reasoning models (even default value)
        assert "temperature" not in result
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()

    def test_knowledge_base_ids_mapping(self):
        """Test knowledge_base_ids mapping to OpenAI tools format."""
        kaapi_params = TextLLMParams(
            model="gpt-4",
            knowledge_base_ids=["vs_abc123", "vs_def456"],
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-4"
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "file_search"
        assert result["tools"][0]["vector_store_ids"] == ["vs_abc123", "vs_def456"]
        assert result["tools"][0]["max_num_results"] == 20  # default
        assert warnings == []

    def test_knowledge_base_with_max_num_results(self):
        """Test knowledge_base_ids with custom max_num_results."""
        kaapi_params = TextLLMParams(
            model="gpt-4",
            knowledge_base_ids=["vs_abc123"],
            max_num_results=50,
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["tools"][0]["max_num_results"] == 50
        assert warnings == []

    def test_complete_parameter_mapping(self):
        """Test mapping all compatible parameters together."""
        kaapi_params = TextLLMParams(
            model="gpt-4o",
            instructions="You are an expert assistant.",
            temperature=0.8,
            knowledge_base_ids=["vs_123"],
            max_num_results=30,
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-4o"
        assert result["instructions"] == "You are an expert assistant."
        assert result["temperature"] == 0.8
        assert result["tools"][0]["type"] == "file_search"
        assert result["tools"][0]["vector_store_ids"] == ["vs_123"]
        assert result["tools"][0]["max_num_results"] == 30
        assert warnings == []

    def test_reasoning_suppressed_for_non_reasoning_models(self):
        """Test that reasoning is suppressed with warning for non-reasoning models."""
        kaapi_params = TextLLMParams(
            model="gpt-4",
            reasoning="high",
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-4"
        assert "reasoning" not in result
        assert len(warnings) == 1
        assert "reasoning" in warnings[0].lower()
        assert "does not support reasoning" in warnings[0]

    def test_temperature_suppressed_for_reasoning_models(self):
        """Test that temperature is suppressed with warning for reasoning models when reasoning is set."""
        kaapi_params = TextLLMParams(
            model="o1",
            temperature=0.7,
            reasoning="high",
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "o1"
        assert result["reasoning"] == {"effort": "high"}
        assert "temperature" not in result
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()
        assert "suppressed" in warnings[0]

    def test_temperature_without_reasoning_for_reasoning_models(self):
        """Test that temperature is suppressed for reasoning models even without explicit reasoning parameter."""
        kaapi_params = TextLLMParams(
            model="o1",
            temperature=0.7,
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "o1"
        assert "temperature" not in result
        assert "reasoning" not in result
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()
        assert "suppressed" in warnings[0]

    def test_minimal_params(self):
        """Test mapping with minimal parameters (only model)."""
        kaapi_params = TextLLMParams(model="gpt-4")

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        # TextLLMParams has default temperature=0.1
        assert result == {"model": "gpt-4", "temperature": 0.1}
        assert warnings == []

    def test_only_knowledge_base_ids(self):
        """Test mapping with only knowledge_base_ids and model."""
        kaapi_params = TextLLMParams(
            model="gpt-4",
            knowledge_base_ids=["vs_xyz"],
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-4"
        assert "tools" in result
        assert result["tools"][0]["vector_store_ids"] == ["vs_xyz"]
        assert warnings == []


class TestMapKaapiToGoogleParams:
    """Test cases for map_kaapi_to_google_params function."""

    def test_basic_model_mapping(self):
        """Test basic model parameter mapping."""
        kaapi_params = TextLLMParams(model="gemini-2.5-pro")

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        # TextLLMParams has default temperature=0.1
        assert result == {"model": "gemini-2.5-pro", "temperature": 0.1}
        assert warnings == []

    def test_instructions_mapping(self):
        """Test instructions parameter mapping."""
        kaapi_params = STTLLMParams(
            model="gemini-2.5-pro",
            instructions="Transcribe this audio accurately.",
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["instructions"] == "Transcribe this audio accurately."
        assert warnings == []

    def test_temperature_mapping(self):
        """Test temperature parameter mapping."""
        kaapi_params = TextLLMParams(
            model="gemini-2.5-pro",
            temperature=0.7,
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["temperature"] == 0.7
        assert warnings == []

    def test_knowledge_base_ids_warning(self):
        """Test that knowledge_base_ids are not supported and generate warning."""
        kaapi_params = TextLLMParams(
            model="gemini-2.5-pro",
            knowledge_base_ids=["vs_abc123"],
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gemini-2.5-pro"
        assert "knowledge_base_ids" not in result
        assert len(warnings) == 1
        assert "knowledge_base_ids" in warnings[0].lower()
        assert "not supported" in warnings[0]

    def test_reasoning_passed_through(self):
        kaapi_params = TextLLMParams(
            model="gemini-2.5-pro",
            reasoning="high",
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["reasoning"] == "high"
        assert len(warnings) == 0

    def test_knowledge_base_ids_unsupported(self):
        kaapi_params = TextLLMParams(
            model="gemini-2.5-pro",
            reasoning="medium",
            knowledge_base_ids=["vs_123"],
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["reasoning"] == "medium"
        assert "knowledge_base_ids" not in result
        assert len(warnings) == 1
        assert "knowledge_base_ids" in warnings[0].lower()


class TestMapKaapiToSarvamParams:
    """Test cases for map_kaapi_to_sarvam_params function."""

    def test_stt_basic_mapping(self):
        """Test basic STT parameter mapping."""
        kaapi_params = STTLLMParams(
            model="saarika:v1",
            input_language="hi-IN",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "saarika:v1"
        assert result["language_code"] == "hi-IN"
        assert result["mode"] == "transcribe"
        assert warnings == []

    def test_stt_auto_language_detection(self):
        """Test STT with auto language detection."""
        kaapi_params = STTLLMParams(
            model="saarika:v1",
            input_language="auto",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "saarika:v1"
        assert result["language_code"] == "unknown"
        assert result["mode"] == "transcribe"
        assert warnings == []

    def test_stt_translate_mode(self):
        """Test STT with translation to English."""
        kaapi_params = STTLLMParams(
            model="saarika:v1",
            input_language="hi-IN",
            output_language="en-IN",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "saarika:v1"
        assert result["language_code"] == "hi-IN"
        assert result["mode"] == "translate"
        assert warnings == []

    def test_stt_same_input_output_language(self):
        """Test STT when input and output languages are the same."""
        kaapi_params = STTLLMParams(
            model="saarika:v1",
            input_language="hi-IN",
            output_language="hi-IN",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["mode"] == "transcribe"
        assert warnings == []

    def test_stt_unsupported_instructions_warning(self):
        """Test that instructions parameter generates warning for STT."""
        kaapi_params = STTLLMParams(
            model="saarika:v1",
            input_language="hi-IN",
            instructions="Please transcribe accurately",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "saarika:v1"
        assert "instructions" not in result
        assert len(warnings) == 1
        assert "instructions" in warnings[0].lower()
        assert "not supported" in warnings[0]

    def test_stt_unsupported_temperature_warning(self):
        """Test that temperature parameter generates warning for STT."""
        kaapi_params = STTLLMParams(
            model="saarika:v1",
            input_language="hi-IN",
            temperature=0.5,
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "saarika:v1"
        assert "temperature" not in result
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()
        assert "not supported" in warnings[0]

    def test_stt_unsupported_response_format_warning(self):
        """Test that response_format parameter generates warning for STT."""
        kaapi_params = STTLLMParams(
            model="saarika:v1",
            input_language="hi-IN",
            response_format="text",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "saarika:v1"
        assert "response_format" not in result
        assert len(warnings) == 1
        assert "response_format" in warnings[0].lower()

    def test_stt_multiple_unsupported_params(self):
        """Test STT with multiple unsupported parameters."""
        kaapi_params = STTLLMParams(
            model="saarika:v1",
            input_language="hi-IN",
            instructions="Transcribe",
            temperature=0.5,
            response_format="text",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "saarika:v1"
        assert "instructions" not in result
        assert "temperature" not in result
        assert "response_format" not in result
        assert len(warnings) == 3

    def test_tts_basic_mapping(self):
        """Test basic TTS parameter mapping."""
        kaapi_params = TTSLLMParams(
            model="bulbul:v1",
            voice="meera",
            language="hi-IN",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "bulbul:v1"
        assert result["speaker"] == "meera"
        assert result["target_language_code"] == "hi-IN"
        assert warnings == []

    def test_tts_with_audio_format(self):
        """Test TTS with custom audio format."""
        kaapi_params = TTSLLMParams(
            model="bulbul:v1",
            voice="meera",
            language="hi-IN",
            response_format="mp3",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "bulbul:v1"
        assert result["speaker"] == "meera"
        assert result["target_language_code"] == "hi-IN"
        assert result["output_audio_codec"] == "mp3"
        assert warnings == []

    def test_tts_default_wav_format(self):
        """Test TTS with default WAV format."""
        kaapi_params = TTSLLMParams(
            model="bulbul:v1",
            voice="arvind",
            language="en-IN",
            response_format="wav",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["output_audio_codec"] == "wav"
        assert warnings == []

    def test_tts_ogg_format(self):
        """Test TTS with OGG format."""
        kaapi_params = TTSLLMParams(
            model="bulbul:v1",
            voice="meera",
            language="hi-IN",
            response_format="ogg",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["output_audio_codec"] == "ogg"
        assert warnings == []

    def test_tts_missing_language(self):
        """Test that missing language returns error for TTS."""
        kaapi_params = {"model": "bulbul:v1", "voice": "meera"}

        result, warnings = map_kaapi_to_sarvam_params(kaapi_params)

        assert result == {}
        assert len(warnings) == 1
        assert "language" in warnings[0].lower()

    def test_missing_model(self):
        """Test that missing model returns error."""
        kaapi_params = {"voice": "meera", "language": "hi-IN"}

        result, warnings = map_kaapi_to_sarvam_params(kaapi_params)

        assert result == {}
        assert len(warnings) == 1
        assert "model" in warnings[0].lower()

    def test_stt_output_language_defaults_to_input(self):
        """Test that output_language defaults to input_language when not provided."""
        kaapi_params = STTLLMParams(
            model="saarika:v1",
            input_language="hi-IN",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["mode"] == "transcribe"
        assert warnings == []


class TestTransformKaapiConfigToNative:
    """Test cases for transform_kaapi_config_to_native function."""

    def test_transform_openai_config(self):
        """Test transformation of Kaapi OpenAI config to native format."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4",
                "temperature": 0.7,
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "openai-native"
        assert result.params["model"] == "gpt-4"
        assert result.params["temperature"] == 0.7
        assert warnings == []

    def test_transform_with_all_params(self):
        """Test transformation with all Kaapi parameters."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
                "instructions": "System prompt here",
                "temperature": 0.5,
                "knowledge_base_ids": ["vs_abc"],
                "max_num_results": 25,
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert result.provider == "openai-native"
        assert result.params["model"] == "gpt-4o"
        assert result.params["instructions"] == "System prompt here"
        assert result.params["temperature"] == 0.5
        assert result.params["tools"][0]["type"] == "file_search"
        assert result.params["tools"][0]["max_num_results"] == 25
        assert warnings == []

    def test_transform_with_reasoning(self):
        """Test transformation with reasoning parameter for reasoning-capable models."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "o1",
                "reasoning": "medium",
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert result.provider == "openai-native"
        assert result.params["model"] == "o1"
        assert result.params["reasoning"] == {"effort": "medium"}
        # Temperature is suppressed for reasoning models (even default value)
        assert "temperature" not in result.params
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()

    def test_transform_with_both_temperature_and_reasoning(self):
        """Test that transformation handles temperature + reasoning intelligently for reasoning models."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "o1",
                "temperature": 0.7,
                "reasoning": "high",
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert result.provider == "openai-native"
        assert result.params["model"] == "o1"
        assert result.params["reasoning"] == {"effort": "high"}
        assert "temperature" not in result.params
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()
        assert "suppressed" in warnings[0]

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported providers raise ValueError."""
        # Note: This would require modifying KaapiCompletionConfig to accept other providers
        # For now, this tests the error handling in the mapper
        # We'll create a mock config that bypasses validation
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.provider = "unsupported-provider"
        mock_config.params = {"model": "some-model"}

        with pytest.raises(ValueError) as exc_info:
            transform_kaapi_config_to_native(mock_config)

        assert "Unsupported provider" in str(exc_info.value)

    def test_transform_preserves_param_structure(self):
        """Test that transformation correctly structures nested parameters."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4",
                "knowledge_base_ids": ["vs_1", "vs_2", "vs_3"],
                "max_num_results": 15,
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        # Verify the nested structure is correct
        assert isinstance(result.params["tools"], list)
        assert isinstance(result.params["tools"][0], dict)
        assert isinstance(result.params["tools"][0]["vector_store_ids"], list)
        assert len(result.params["tools"][0]["vector_store_ids"]) == 3
        assert warnings == []

    def test_transform_google_config(self):
        """Test transformation of Kaapi Google AI config to native format."""
        kaapi_config = KaapiCompletionConfig(
            provider="google",
            type="stt",
            params={
                "model": "gemini-2.5-pro",
                "instructions": "Transcribe accurately",
                "temperature": 0.2,
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "google-native"
        assert result.params["model"] == "gemini-2.5-pro"
        assert result.params["instructions"] == "Transcribe accurately"
        assert result.params["temperature"] == 0.2
        assert warnings == []

    def test_transform_google_with_unsupported_params(self):
        kaapi_config = KaapiCompletionConfig(
            provider="google",
            type="text",
            params={
                "model": "gemini-2.5-pro",
                "knowledge_base_ids": ["vs_123"],
                "reasoning": "high",
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert result.provider == "google-native"
        assert result.params["model"] == "gemini-2.5-pro"
        assert result.params["reasoning"] == "high"
        assert "knowledge_base_ids" not in result.params
        assert len(warnings) == 1

    def test_transform_sarvamai_stt_config(self):
        """Test transformation of Kaapi SarvamAI STT config to native format."""
        kaapi_config = KaapiCompletionConfig(
            provider="sarvamai",
            type="stt",
            params={
                "model": "saarika:v1",
                "input_language": "hi-IN",
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "sarvamai-native"
        assert result.type == "stt"
        assert result.params["model"] == "saarika:v1"
        assert result.params["language_code"] == "hi-IN"
        assert result.params["mode"] == "transcribe"
        assert warnings == []

    def test_transform_sarvamai_tts_config(self):
        """Test transformation of Kaapi SarvamAI TTS config to native format."""
        kaapi_config = KaapiCompletionConfig(
            provider="sarvamai",
            type="tts",
            params={
                "model": "bulbul:v1",
                "voice": "meera",
                "language": "hi-IN",
                "response_format": "mp3",
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "sarvamai-native"
        assert result.type == "tts"
        assert result.params["model"] == "bulbul:v1"
        assert result.params["speaker"] == "meera"
        assert result.params["target_language_code"] == "hi-IN"
        assert result.params["output_audio_codec"] == "mp3"
        assert warnings == []

    def test_transform_sarvamai_stt_with_unsupported_params(self):
        """Test SarvamAI STT transformation with unsupported parameters."""
        kaapi_config = KaapiCompletionConfig(
            provider="sarvamai",
            type="stt",
            params={
                "model": "saarika:v1",
                "input_language": "hi-IN",
                "instructions": "Transcribe carefully",
                "temperature": 0.5,
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert result.provider == "sarvamai-native"
        assert result.params["model"] == "saarika:v1"
        assert "instructions" not in result.params
        assert "temperature" not in result.params
        assert len(warnings) == 2
        assert any("instructions" in w.lower() for w in warnings)
        assert any("temperature" in w.lower() for w in warnings)
