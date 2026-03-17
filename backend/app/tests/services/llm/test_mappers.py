"""
Unit tests for LLM parameter mapping functions.

Tests the transformation of Kaapi-abstracted parameters to provider-native formats.
Covers real-world scenarios, edge cases, and provider-specific requirements.
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
    map_kaapi_to_elevenlabs_params,
    bcp47_to_elevenlabs_lang,
    voice_to_id,
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
            max_num_results=50,
        )

        result, warnings = map_kaapi_to_openai_params(
            kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-4"
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "file_search"
        assert result["tools"][0]["vector_store_ids"] == ["vs_abc123", "vs_def456"]
        assert result["tools"][0]["max_num_results"] == 50
        assert warnings == []

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


class TestMapKaapiToGoogleParams:
    """Test cases for map_kaapi_to_google_params function with completion_type."""

    def test_text_completion_basic(self):
        """Test basic text completion parameter mapping."""
        kaapi_params = TextLLMParams(model="gemini-2.5-pro", temperature=0.7)

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="text"
        )

        assert result == {"model": "gemini-2.5-pro", "temperature": 0.7}
        assert warnings == []

    def test_text_completion_with_reasoning(self):
        """Test text completion with reasoning parameter."""
        kaapi_params = TextLLMParams(
            model="gemini-2.5-pro", reasoning="high", temperature=0.5
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="text"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["reasoning"] == "high"
        assert result["temperature"] == 0.5
        assert warnings == []

    def test_text_completion_knowledge_base_unsupported(self):
        """Test that knowledge_base_ids generate warning for Google AI."""
        kaapi_params = TextLLMParams(
            model="gemini-2.5-pro", knowledge_base_ids=["vs_abc123"]
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="text"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert "knowledge_base_ids" not in result
        assert len(warnings) == 1
        assert "knowledge_base_ids" in warnings[0].lower()
        assert "not supported" in warnings[0]

    def test_stt_completion_with_instructions(self):
        """Test STT completion with instructions parameter."""
        kaapi_params = STTLLMParams(
            model="gemini-2.5-pro", instructions="Transcribe accurately"
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["instructions"] == "Transcribe accurately"
        assert warnings == []

    def test_tts_completion_with_voice(self):
        """Test TTS completion with voice and language parameters."""
        kaapi_params = TTSLLMParams(
            model="gemini-2.5-pro", voice="en-US-Journey-D", language="en-US"
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["voice"] == "en-US-Journey-D"
        assert result["language"] == "en-US"
        assert warnings == []

    def test_unsupported_completion_type(self):
        """Test that unsupported completion types return error."""
        kaapi_params = {"model": "gemini-2.5-pro"}

        result, warnings = map_kaapi_to_google_params(
            kaapi_params, completion_type="invalid"
        )

        assert result == {}
        assert len(warnings) == 1
        assert "Unsupported completion type" in warnings[0]


class TestMapKaapiToSarvamParams:
    """Test cases for map_kaapi_to_sarvam_params function with real-world scenarios."""

    # STT Tests
    def test_stt_basic_with_saarika_model(self):
        """Test STT with saarika model (mode should NOT be set)."""
        kaapi_params = STTLLMParams(model="saarika:v2.5", input_language="hi-IN")

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model"] == "saarika:v2.5"
        assert result["language_code"] == "hi-IN"
        # mode should NOT be set for saarika models
        assert "mode" not in result
        assert warnings == []

    def test_stt_with_saaras_model_transcribe_mode(self):
        """Test STT with saaras:v3 model (mode SHOULD be set)."""
        kaapi_params = STTLLMParams(model="saaras:v3", input_language="hi-IN")

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model"] == "saaras:v3"
        assert result["language_code"] == "hi-IN"
        # mode should be set for saaras:v3
        assert result["mode"] == "transcribe"
        assert warnings == []

    def test_stt_with_saaras_model_translate_mode(self):
        """Test STT with saaras:v3 model in translate mode."""
        kaapi_params = STTLLMParams(
            model="saaras:v3", input_language="hi-IN", output_language="en-IN"
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model"] == "saaras:v3"
        assert result["language_code"] == "hi-IN"
        assert result["mode"] == "translate"
        assert warnings == []

    def test_stt_auto_language_detection(self):
        """Test STT with auto language detection."""
        kaapi_params = STTLLMParams(model="saarika:v2.5", input_language="auto")

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model"] == "saarika:v2.5"
        assert result["language_code"] == "unknown"
        assert warnings == []

    def test_stt_missing_input_language_defaults_to_unknown(self):
        """Test STT without input_language defaults to 'unknown' for auto-detection."""
        kaapi_params = {"model": "saarika:v2.5"}

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params, completion_type="stt"
        )

        assert result["model"] == "saarika:v2.5"
        # Should default to unknown for auto-detection
        assert result["language_code"] == "unknown"
        assert warnings == []

    def test_stt_unsupported_params_generate_warnings(self):
        """Test that unsupported STT parameters generate warnings."""
        kaapi_params = STTLLMParams(
            model="saarika:v2.5",
            input_language="hi-IN",
            instructions="Transcribe carefully",
            temperature=0.5,
            response_format="text",
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model"] == "saarika:v2.5"
        assert "instructions" not in result
        assert "temperature" not in result
        assert "response_format" not in result
        assert len(warnings) == 3
        assert any("instructions" in w.lower() for w in warnings)
        assert any("temperature" in w.lower() for w in warnings)
        assert any("response_format" in w.lower() for w in warnings)

    # TTS Tests
    def test_tts_basic_with_all_required_params(self):
        """Test TTS with all required parameters."""
        kaapi_params = TTSLLMParams(model="bulbul:v3", voice="Shubh", language="hi-IN")

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["model"] == "bulbul:v3"
        assert result["speaker"] == "Shubh"
        assert result["target_language_code"] == "hi-IN"
        assert warnings == []

    def test_tts_missing_language_returns_error(self):
        """Test that missing language parameter returns error."""
        kaapi_params = {"model": "bulbul:v3", "voice": "Shubh"}

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params, completion_type="tts"
        )

        assert result == {}
        assert len(warnings) == 1
        assert "language" in warnings[0].lower()

    def test_tts_optional_voice_parameter(self):
        """Test TTS without voice parameter (should use API default)."""
        kaapi_params = {"model": "bulbul:v3", "language": "hi-IN"}

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params, completion_type="tts"
        )

        assert result["model"] == "bulbul:v3"
        assert result["target_language_code"] == "hi-IN"
        # speaker should not be set if not provided (API will use default)
        assert "speaker" not in result
        assert warnings == []

    def test_tts_audio_format_mp3(self):
        """Test TTS with MP3 audio format."""
        kaapi_params = TTSLLMParams(
            model="bulbul:v3", voice="Anushka", language="hi-IN", response_format="mp3"
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["output_audio_codec"] == "mp3"
        assert warnings == []

    def test_tts_audio_format_ogg_maps_to_opus(self):
        """Test TTS with OGG format maps to OPUS (closest supported)."""
        kaapi_params = TTSLLMParams(
            model="bulbul:v3", voice="Shubh", language="hi-IN", response_format="ogg"
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        # OGG should map to OPUS (closest match)
        assert result["output_audio_codec"] == "opus"
        assert warnings == []

    def test_tts_audio_format_wav(self):
        """Test TTS with WAV audio format."""
        kaapi_params = TTSLLMParams(
            model="bulbul:v3", voice="Shubh", language="hi-IN", response_format="wav"
        )

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["output_audio_codec"] == "wav"
        assert warnings == []

    # Error Cases
    def test_missing_model_returns_error(self):
        """Test that missing model parameter returns error."""
        kaapi_params = {"voice": "Shubh", "language": "hi-IN"}

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params, completion_type="tts"
        )

        assert result == {}
        assert len(warnings) == 1
        assert "model" in warnings[0].lower()

    def test_unsupported_completion_type(self):
        """Test that unsupported completion types return error."""
        kaapi_params = {"model": "saarika:v2.5"}

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params, completion_type="invalid"
        )

        assert result == {}
        assert len(warnings) == 1
        assert "Unsupported completion type" in warnings[0]


class TestMapKaapiToElevenlabsParams:
    """Test cases for map_kaapi_to_elevenlabs_params function."""

    # STT Tests
    def test_stt_basic_with_language(self):
        """Test STT with language code."""
        kaapi_params = STTLLMParams(
            model="scribe_v2", input_language="hi-IN", temperature=0.3
        )

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model_id"] == "scribe_v2"
        assert result["language_code"] == "hi"  # BCP-47 conversion
        assert result["temperature"] == 0.3
        assert warnings == []

    def test_stt_auto_language_detection(self):
        """Test STT with auto language detection."""
        kaapi_params = STTLLMParams(model="scribe_v2", input_language="auto")

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model_id"] == "scribe_v2"
        assert result["language_code"] is None
        assert warnings == []

    def test_stt_missing_language_defaults_to_unknown(self):
        """Test STT without language defaults to None for auto-detection."""
        kaapi_params = {"model": "scribe_v2"}

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params, completion_type="stt"
        )

        assert result["model_id"] == "scribe_v2"
        # No language_code should be set when not provided
        assert "language_code" not in result
        assert warnings == []

    def test_stt_unsupported_language_generates_warning(self):
        """Test STT with unsupported language generates warning."""
        kaapi_params = STTLLMParams(model="scribe_v2", input_language="fr-FR")

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model_id"] == "scribe_v2"
        assert len(warnings) == 1
        assert "Unsupported language" in warnings[0]
        assert "auto-detect" in warnings[0]

    def test_stt_output_language_translation_warning(self):
        """Test STT with different output language generates warning."""
        kaapi_params = STTLLMParams(
            model="scribe_v2", input_language="hi-IN", output_language="en-IN"
        )

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model_id"] == "scribe_v2"
        assert len(warnings) == 1
        assert "output_language" in warnings[0].lower()
        assert "translation" in warnings[0].lower()

    def test_stt_unsupported_instructions_warning(self):
        """Test STT with instructions generates warning."""
        kaapi_params = STTLLMParams(
            model="scribe_v2",
            input_language="hi-IN",
            instructions="Transcribe accurately",
        )

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model_id"] == "scribe_v2"
        assert "instructions" not in result
        assert len(warnings) == 1
        assert "instructions" in warnings[0].lower()

    # TTS Tests
    def test_tts_basic_with_voice_and_language(self):
        """Test TTS with voice and language."""
        kaapi_params = TTSLLMParams(
            model="eleven_turbo_v2", voice="Sarah", language="en-IN"
        )

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["model_id"] == "eleven_turbo_v2"
        assert result["voice_id"] == "EXAVITQu4vr4xnSDxMaL"  # Sarah's ID
        assert result["language_code"] == "en"
        assert warnings == []

    def test_tts_missing_voice_returns_error(self):
        """Test that missing voice parameter returns error."""
        kaapi_params = {"model": "eleven_turbo_v2", "language": "en-IN"}

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params, completion_type="tts"
        )

        assert result == {}
        assert len(warnings) == 1
        assert "voice" in warnings[0].lower()

    def test_tts_unsupported_voice_returns_error(self):
        """Test that unsupported voice returns error."""
        kaapi_params = TTSLLMParams(
            model="eleven_turbo_v2", voice="InvalidVoice", language="en-IN"
        )

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result == {}
        assert len(warnings) == 1
        assert "Unsupported voice" in warnings[0]

    def test_tts_optional_language_parameter(self):
        """Test TTS without language (should be optional)."""
        kaapi_params = {"model": "eleven_turbo_v2", "voice": "Sarah"}

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params, completion_type="tts"
        )

        assert result["model_id"] == "eleven_turbo_v2"
        assert result["voice_id"] == "EXAVITQu4vr4xnSDxMaL"
        # language_code should not be set if language not provided
        assert "language_code" not in result
        assert warnings == []

    def test_tts_unsupported_language_generates_warning(self):
        """Test TTS with unsupported language generates warning."""
        kaapi_params = TTSLLMParams(
            model="eleven_turbo_v2", voice="Sarah", language="fr-FR"
        )

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["model_id"] == "eleven_turbo_v2"
        assert result["voice_id"] == "EXAVITQu4vr4xnSDxMaL"
        assert "language_code" not in result
        assert len(warnings) == 1
        assert "Unsupported language" in warnings[0]

    def test_tts_audio_format_mp3(self):
        """Test TTS with MP3 format."""
        kaapi_params = TTSLLMParams(
            model="eleven_turbo_v2",
            voice="George",
            language="en-IN",
            response_format="mp3",
        )

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["output_format"] == "mp3_44100_128"
        assert warnings == []

    def test_tts_audio_format_wav(self):
        """Test TTS with WAV format."""
        kaapi_params = TTSLLMParams(
            model="eleven_turbo_v2",
            voice="Callum",
            language="en-IN",
            response_format="wav",
        )

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["output_format"] == "wav_24000"
        assert warnings == []

    def test_tts_audio_format_ogg_maps_to_opus(self):
        """Test TTS with OGG format maps to OPUS."""
        kaapi_params = TTSLLMParams(
            model="eleven_turbo_v2",
            voice="Liam",
            language="en-IN",
            response_format="ogg",
        )

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        # OGG maps to OPUS for ElevenLabs
        assert result["output_format"] == "opus_48000_128"
        assert warnings == []

    def test_tts_all_supported_voices(self):
        """Test TTS with all supported voices map correctly."""
        voices = {
            "Sarah": "EXAVITQu4vr4xnSDxMaL",
            "George": "JBFqnCBsd6RMkjVDRZzb",
            "Callum": "N2lVS1w4EtoT3dr4eOWO",
            "Liam": "TX3LPaxmHKxFdv7VOQHJ",
        }

        for voice_name, expected_id in voices.items():
            kaapi_params = TTSLLMParams(
                model="eleven_turbo_v2", voice=voice_name, language="en-IN"
            )

            result, warnings = map_kaapi_to_elevenlabs_params(
                kaapi_params.model_dump(exclude_none=True), completion_type="tts"
            )

            assert result["voice_id"] == expected_id
            assert warnings == []

    # Error Cases
    def test_missing_model_returns_error(self):
        """Test that missing model returns error."""
        kaapi_params = {"voice": "Sarah", "language": "en-IN"}

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params, completion_type="tts"
        )

        assert result == {}
        assert len(warnings) == 1
        assert "model" in warnings[0].lower()

    def test_unsupported_completion_type(self):
        """Test that unsupported completion types return error."""
        kaapi_params = {"model": "eleven_turbo_v2"}

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params, completion_type="invalid"
        )

        assert result == {}
        assert len(warnings) == 1
        assert "Unsupported completion type" in warnings[0]


class TestBCP47ToElevenlabsLang:
    """Test BCP-47 language code conversion for ElevenLabs."""

    def test_supported_indian_languages(self):
        """Test conversion of supported Indian languages."""
        test_cases = {
            "en-IN": "en",
            "hi-IN": "hi",
            "bn-IN": "bn",
            "ta-IN": "ta",
            "te-IN": "te",
            "mr-IN": "mr",
            "gu-IN": "gu",
            "kn-IN": "kn",
            "ml-IN": "ml",
            "pa-IN": "pa",
        }

        for bcp47, expected in test_cases.items():
            result = bcp47_to_elevenlabs_lang(bcp47)
            assert result == expected

    def test_unsupported_language_returns_none(self):
        """Test that unsupported languages return None."""
        assert bcp47_to_elevenlabs_lang("fr-FR") is None
        assert bcp47_to_elevenlabs_lang("de-DE") is None
        assert bcp47_to_elevenlabs_lang("invalid") is None


class TestVoiceToId:
    """Test voice name to ID conversion for ElevenLabs."""

    def test_supported_voices(self):
        """Test conversion of supported voice names."""
        test_cases = {
            "Sarah": "EXAVITQu4vr4xnSDxMaL",
            "George": "JBFqnCBsd6RMkjVDRZzb",
            "Callum": "N2lVS1w4EtoT3dr4eOWO",
            "Liam": "TX3LPaxmHKxFdv7VOQHJ",
        }

        for voice_name, expected_id in test_cases.items():
            result = voice_to_id(voice_name)
            assert result == expected_id

    def test_unsupported_voice_returns_none(self):
        """Test that unsupported voices return None."""
        assert voice_to_id("InvalidVoice") is None
        assert voice_to_id("UnknownSpeaker") is None


class TestTransformKaapiConfigToNative:
    """Test end-to-end transformation with completion_type parameter."""

    def test_transform_elevenlabs_tts_config(self):
        """Test transformation of ElevenLabs TTS config."""
        kaapi_config = KaapiCompletionConfig(
            provider="elevenlabs",
            type="tts",
            params={
                "model": "eleven_turbo_v2",
                "voice": "Sarah",
                "language": "en-IN",
                "response_format": "mp3",
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "elevenlabs-native"
        assert result.type == "tts"
        assert result.params["model_id"] == "eleven_turbo_v2"
        assert result.params["voice_id"] == "EXAVITQu4vr4xnSDxMaL"
        assert result.params["language_code"] == "en"
        assert result.params["output_format"] == "mp3_44100_128"
        assert warnings == []

    def test_transform_elevenlabs_stt_config(self):
        """Test transformation of ElevenLabs STT config."""
        kaapi_config = KaapiCompletionConfig(
            provider="elevenlabs",
            type="stt",
            params={
                "model": "scribe_v2",
                "input_language": "hi-IN",
                "temperature": 0.3,
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "elevenlabs-native"
        assert result.type == "stt"
        assert result.params["model_id"] == "scribe_v2"
        assert result.params["language_code"] == "hi"
        assert result.params["temperature"] == 0.3
        assert warnings == []

    def test_transform_sarvamai_stt_with_saaras_model(self):
        """Test transformation of SarvamAI STT with saaras:v3 model."""
        kaapi_config = KaapiCompletionConfig(
            provider="sarvamai",
            type="stt",
            params={
                "model": "saaras:v3",
                "input_language": "hi-IN",
                "output_language": "en-IN",
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "sarvamai-native"
        assert result.type == "stt"
        assert result.params["model"] == "saaras:v3"
        assert result.params["language_code"] == "hi-IN"
        # mode should be set for saaras:v3
        assert result.params["mode"] == "translate"
        assert warnings == []

    def test_transform_sarvamai_stt_with_saarika_model(self):
        """Test transformation of SarvamAI STT with saarika model (no mode)."""
        kaapi_config = KaapiCompletionConfig(
            provider="sarvamai",
            type="stt",
            params={"model": "saarika:v2.5", "input_language": "hi-IN"},
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "sarvamai-native"
        assert result.type == "stt"
        assert result.params["model"] == "saarika:v2.5"
        assert result.params["language_code"] == "hi-IN"
        # mode should NOT be set for saarika models
        assert "mode" not in result.params
        assert warnings == []

    def test_transform_sarvamai_tts_with_optional_voice(self):
        """Test transformation of SarvamAI TTS without voice (using API default)."""
        kaapi_config = KaapiCompletionConfig(
            provider="sarvamai",
            type="tts",
            params={"model": "bulbul:v3", "language": "hi-IN"},
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "sarvamai-native"
        assert result.type == "tts"
        assert result.params["model"] == "bulbul:v3"
        assert result.params["target_language_code"] == "hi-IN"
        # speaker should not be set (will use API default)
        assert "speaker" not in result.params
        assert warnings == []

    def test_transform_google_text_completion(self):
        """Test transformation of Google text completion."""
        kaapi_config = KaapiCompletionConfig(
            provider="google",
            type="text",
            params={
                "model": "gemini-2.5-pro",
                "temperature": 0.7,
                "reasoning": "high",
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "google-native"
        assert result.type == "text"
        assert result.params["model"] == "gemini-2.5-pro"
        assert result.params["temperature"] == 0.7
        assert result.params["reasoning"] == "high"
        assert warnings == []

    def test_transform_google_stt_completion(self):
        """Test transformation of Google STT completion."""
        kaapi_config = KaapiCompletionConfig(
            provider="google",
            type="stt",
            params={"model": "gemini-2.5-pro", "instructions": "Transcribe accurately"},
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "google-native"
        assert result.type == "stt"
        assert result.params["model"] == "gemini-2.5-pro"
        assert result.params["instructions"] == "Transcribe accurately"
        assert warnings == []

    def test_transform_google_tts_completion(self):
        """Test transformation of Google TTS completion."""
        kaapi_config = KaapiCompletionConfig(
            provider="google",
            type="tts",
            params={
                "model": "gemini-2.5-pro",
                "voice": "en-US-Journey-D",
                "language": "en-US",
            },
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "google-native"
        assert result.type == "tts"
        assert result.params["model"] == "gemini-2.5-pro"
        assert result.params["voice"] == "en-US-Journey-D"
        assert result.params["language"] == "en-US"
        assert warnings == []
