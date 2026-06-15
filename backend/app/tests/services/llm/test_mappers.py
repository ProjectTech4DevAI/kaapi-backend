"""
Unit tests for LLM parameter mapping functions.

Tests the transformation of Kaapi-abstracted parameters to provider-native formats.
Covers real-world scenarios, edge cases, and provider-specific requirements.
"""

from sqlmodel import Session

from app.models.llm.request import (
    KaapiCompletionConfig,
    NativeCompletionConfig,
    STTLLMParams,
    TextLLMParams,
    TTSLLMParams,
)
from app.services.llm.mappers import (
    bcp47_to_elevenlabs_lang,
    map_kaapi_to_anthropic_params,
    map_kaapi_to_elevenlabs_params,
    map_kaapi_to_google_params,
    map_kaapi_to_openai_params,
    map_kaapi_to_sarvam_params,
    transform_kaapi_config_to_native,
    voice_to_id,
)
import pytest


class TestMapKaapiToOpenAIParams:
    """Test cases for map_kaapi_to_openai_params function."""

    def test_basic_model_mapping(self, db: Session):
        """Test basic model parameter mapping."""
        kaapi_params = TextLLMParams(model="gpt-4o")

        result, warnings = map_kaapi_to_openai_params(
            session=db, kaapi_params=kaapi_params.model_dump(exclude_none=True)
        )

        # TextLLMParams has default temperature=0.1
        assert result == {"model": "gpt-4o", "temperature": 0.1}
        assert warnings == []

    def test_reasoning_mapping_for_reasoning_models(self, db: Session):
        """Test reasoning parameter mapping to OpenAI format for reasoning-capable models."""
        kaapi_params = TextLLMParams(
            model="gpt-5",
            reasoning="high",
        )

        result, warnings = map_kaapi_to_openai_params(
            session=db, kaapi_params=kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-5"
        assert result["reasoning"] == {"effort": "high"}
        # Temperature is suppressed for reasoning models (even default value)
        assert "temperature" not in result
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()

    def test_knowledge_base_ids_mapping(self, db: Session):
        """Test knowledge_base_ids mapping to OpenAI tools format."""
        kaapi_params = TextLLMParams(
            model="gpt-4o",
            knowledge_base_ids=["vs_abc123", "vs_def456"],
            max_num_results=50,
        )

        result, warnings = map_kaapi_to_openai_params(
            session=db, kaapi_params=kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-4o"
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "file_search"
        assert result["tools"][0]["vector_store_ids"] == ["vs_abc123", "vs_def456"]
        assert result["tools"][0]["max_num_results"] == 50
        assert warnings == []

    def test_temperature_suppressed_for_reasoning_models(self, db: Session):
        """Test that temperature is suppressed with warning for reasoning models when reasoning is set."""
        kaapi_params = TextLLMParams(
            model="gpt-5",
            temperature=0.7,
            reasoning="high",
        )

        result, warnings = map_kaapi_to_openai_params(
            session=db, kaapi_params=kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-5"
        assert result["reasoning"] == {"effort": "high"}
        assert "temperature" not in result
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()
        assert "suppressed" in warnings[0]

    def test_reasoning_suppressed_for_non_reasoning_models(self, db: Session):
        """Test that reasoning is suppressed with warning for non-reasoning models."""
        kaapi_params = TextLLMParams(
            model="gpt-4o",
            reasoning="high",
        )

        result, warnings = map_kaapi_to_openai_params(
            session=db, kaapi_params=kaapi_params.model_dump(exclude_none=True)
        )

        assert result["model"] == "gpt-4o"
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
        assert result["input_language"] == "auto"  # Default from STTLLMParams
        assert warnings == []

    def test_tts_completion_with_voice(self):
        """Test TTS completion with voice and language parameters."""
        kaapi_params = TTSLLMParams(
            model="gemini-2.5-pro",
            voice="Orus",
            language="en-IN",  # Use supported BCP-47 locale
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["voice"] == "Orus"
        assert (
            result["language"] == "en"
        )  # Mapped from en-IN to en via BCP47_LOCALE_TO_GEMINI_LANG
        assert result["response_format"] == "wav"  # Default
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

    def test_tts_empty_params_uses_smart_defaults(self):
        """Test TTS with empty params uses smart defaults (voice, response_format)."""
        # Simulate empty params after exclude_none=True strips None values
        kaapi_params = {"model": "gemini-2.5-pro"}

        result, warnings = map_kaapi_to_google_params(
            kaapi_params, completion_type="tts"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["voice"] == "Kore"  # DEFAULT_TTS_VOICE
        assert result["response_format"] == "wav"  # Default
        assert "language" not in result  # None = auto-detect, not set
        assert warnings == []

    def test_stt_empty_params_uses_smart_defaults(self):
        """Test STT with empty params uses smart defaults (input_language=auto)."""
        kaapi_params = {"model": "gemini-2.5-pro"}

        result, warnings = map_kaapi_to_google_params(
            kaapi_params, completion_type="stt"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["input_language"] == "auto"  # Default
        assert "output_language" not in result  # Optional, not set
        assert "instructions" not in result  # Optional, not set
        assert warnings == []

    def test_tts_language_missing_uses_auto_detect(self):
        """Test TTS with missing language parameter (auto-detect behavior)."""
        kaapi_params = TTSLLMParams(
            model="gemini-2.5-pro", voice="Orus", response_format="mp3"
        )
        # language is None by default, gets stripped by exclude_none=True

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["voice"] == "Orus"
        assert result["response_format"] == "mp3"
        assert "language" not in result  # Not set = auto-detect
        assert warnings == []

    def test_tts_unsupported_language_warns_and_auto_detects(self):
        """Test TTS with unsupported language generates warning and uses auto-detect."""
        kaapi_params = TTSLLMParams(
            model="gemini-2.5-pro",
            voice="Kore",
            language="fr-FR",  # French not in BCP47_LOCALE_TO_GEMINI_LANG
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["voice"] == "Kore"
        assert result["response_format"] == "wav"  # Default
        assert "language" not in result  # Not set, falls back to auto-detect
        assert len(warnings) == 1
        assert "Unsupported language 'fr-FR'" in warnings[0]
        assert "auto-detect" in warnings[0]

    def test_tts_supported_language_maps_correctly(self):
        """Test TTS with supported BCP-47 language maps to Gemini language code."""
        kaapi_params = TTSLLMParams(
            model="gemini-2.5-pro", voice="Kore", language="hi-IN"
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="tts"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["voice"] == "Kore"
        assert (
            result["language"] == "hi"
        )  # BCP47_LOCALE_TO_GEMINI_LANG maps hi-IN -> hi
        assert result["response_format"] == "wav"  # Default
        assert warnings == []

    def test_stt_with_input_and_output_language(self):
        """Test STT with both input and output language (translation scenario)."""
        kaapi_params = STTLLMParams(
            model="gemini-2.5-pro",
            input_language="hi-IN",
            output_language="en-IN",
        )

        result, warnings = map_kaapi_to_google_params(
            kaapi_params.model_dump(exclude_none=True), completion_type="stt"
        )

        assert result["model"] == "gemini-2.5-pro"
        assert result["input_language"] == "hi-IN"
        assert result["output_language"] == "en-IN"
        assert warnings == []


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

    # Error / fallback cases
    def test_missing_model_falls_back_to_default(self):
        """Missing model falls back to DEFAULT_SARVAM_TTS_MODEL without warnings."""
        kaapi_params = {"voice": "Shubh", "language": "hi-IN"}

        result, warnings = map_kaapi_to_sarvam_params(
            kaapi_params, completion_type="tts"
        )

        assert result["model"] == "bulbul:v3"
        assert result["speaker"] == "Shubh"
        assert result["target_language_code"] == "hi-IN"
        assert warnings == []

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

    # Error / fallback cases
    def test_missing_model_falls_back_to_default(self):
        """Missing model falls back to DEFAULT_ELEVENLABS_TTS_MODEL without warnings."""
        kaapi_params = {"voice": "Sarah", "language": "en-IN"}

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params, completion_type="tts"
        )

        assert result["model_id"] == "eleven_v3"
        assert result["voice_id"] == "EXAVITQu4vr4xnSDxMaL"
        assert result["language_code"] == "en"
        assert warnings == []

    def test_unsupported_completion_type(self):
        """Test that unsupported completion types return error."""
        kaapi_params = {"model": "eleven_turbo_v2"}

        result, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_params, completion_type="invalid"
        )

        assert result == {}
        assert len(warnings) == 1
        assert "Unsupported completion type" in warnings[0]


class TestMapKaapiToAnthropicParams:
    """Test cases for map_kaapi_to_anthropic_params."""

    def test_full_text_completion_params_mapped(self):
        """Real-world text-completion payload: every supported Kaapi field
        maps to its Anthropic equivalent, no warnings."""
        kaapi_params = {
            "model": "claude-sonnet-4-6",
            "instructions": "You are a helpful assistant.",
            "temperature": 0.4,
            "top_p": 0.9,
            "max_output_tokens": 1024,
        }

        result, warnings = map_kaapi_to_anthropic_params(kaapi_params)

        assert result == {
            "model": "claude-sonnet-4-6",
            "system": "You are a helpful assistant.",
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 1024,
        }
        assert warnings == []

    def test_missing_model_falls_back_to_default(self):
        """Anthropic requires model — provider falls back to the centralised
        default when caller omits it."""
        result, warnings = map_kaapi_to_anthropic_params({})

        assert result == {"model": "claude-sonnet-4-6"}
        assert warnings == []

    def test_max_output_tokens_renamed_to_max_tokens(self):
        """Kaapi calls it max_output_tokens; Anthropic Messages API calls it
        max_tokens. The rename is the contract — protect against drift."""
        result, _ = map_kaapi_to_anthropic_params(
            {"model": "claude-sonnet-4-6", "max_output_tokens": 256}
        )
        assert "max_tokens" in result
        assert "max_output_tokens" not in result
        assert result["max_tokens"] == 256

    def test_unsupported_knowledge_base_emits_warning_and_drops_field(self):
        """Anthropic has no managed vector store, so we drop knowledge_base_ids
        and surface a warning the caller can show to users."""
        result, warnings = map_kaapi_to_anthropic_params(
            {
                "model": "claude-sonnet-4-6",
                "knowledge_base_ids": ["kb_1", "kb_2"],
            }
        )

        assert "knowledge_base_ids" not in result
        assert len(warnings) == 1
        assert "knowledge_base_ids" in warnings[0]

    def test_reasoning_effort_summary_collapsed_into_single_warning(self):
        """Any of reasoning/effort/summary triggers the same advisory; only
        one warning is emitted regardless of how many are supplied."""
        result, warnings = map_kaapi_to_anthropic_params(
            {
                "model": "claude-sonnet-4-6",
                "reasoning": "high",
                "effort": "medium",
                "summary": "concise",
            }
        )

        assert "reasoning" not in result
        assert "effort" not in result
        assert "summary" not in result
        assert len(warnings) == 1
        assert "reasoning" in warnings[0].lower()

    def test_temperature_zero_is_preserved(self):
        """0.0 is a valid temperature — guard against truthy-check bugs that
        would drop it as if it were None."""
        result, _ = map_kaapi_to_anthropic_params(
            {"model": "claude-sonnet-4-6", "temperature": 0.0}
        )
        assert result["temperature"] == 0.0


class TestTransformGoogleVertexRouting:
    """Routing contract for the ``google`` provider (which executes via
    Vertex AI). Text completions are explicitly rejected — they must go
    through the ``google-aistudio`` provider."""

    def test_text_completion_is_rejected(self, db: Session):
        """``google`` is audio-only (Vertex STT/TTS) — text completions
        must be routed through ``google-aistudio``."""
        kaapi_config = KaapiCompletionConfig(
            provider="google",
            type="text",
            params={"model": "gemini-2.5-pro"},
        )

        with pytest.raises(ValueError) as exc_info:
            transform_kaapi_config_to_native(session=db, kaapi_config=kaapi_config)

        msg = str(exc_info.value)
        assert "google" in msg
        assert "text" in msg
        assert "google-aistudio" in msg  # hints the caller toward the right provider

    def test_unsupported_language_emits_warning(self, db: Session):
        """Languages not in BCP47_LOCALE_TO_GEMINI_LANG fall back to auto-detect
        and surface a warning, rather than silently being dropped."""
        kaapi_config = KaapiCompletionConfig(
            provider="google",
            type="tts",
            params={
                "model": "gemini-2.5-flash-preview-tts",
                "language": "xx-YY",  # unsupported
            },
        )

        native_config, warnings = transform_kaapi_config_to_native(
            session=db, kaapi_config=kaapi_config
        )

        assert native_config.provider == "google-native"
        assert "language" not in native_config.params  # dropped
        assert len(warnings) == 1
        assert "xx-YY" in warnings[0]


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

    def test_transform_elevenlabs_tts_config(self, db: Session):
        """Test transformation of ElevenLabs TTS config."""
        kaapi_config = KaapiCompletionConfig(
            provider="elevenlabs",
            type="tts",
            params={
                "model": "eleven_v3",  # Updated to match current SUPPORTED_MODELS
                "voice": "Sarah",
                "language": "en-IN",
                "response_format": "mp3",
            },
        )

        result, warnings = transform_kaapi_config_to_native(
            session=db, kaapi_config=kaapi_config
        )

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "elevenlabs-native"
        assert result.type == "tts"
        assert result.params["model_id"] == "eleven_v3"  # Updated
        assert result.params["voice_id"] == "EXAVITQu4vr4xnSDxMaL"
        assert result.params["language_code"] == "en"
        assert result.params["output_format"] == "mp3_44100_128"
        assert warnings == []

    def test_transform_elevenlabs_stt_config(self, db: Session):
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

        result, warnings = transform_kaapi_config_to_native(
            session=db, kaapi_config=kaapi_config
        )

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "elevenlabs-native"
        assert result.type == "stt"
        assert result.params["model_id"] == "scribe_v2"
        assert result.params["language_code"] == "hi"
        assert result.params["temperature"] == 0.3
        assert warnings == []

    def test_transform_sarvamai_stt_with_saaras_model(self, db: Session):
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

        result, warnings = transform_kaapi_config_to_native(
            session=db, kaapi_config=kaapi_config
        )

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "sarvamai-native"
        assert result.type == "stt"
        assert result.params["model"] == "saaras:v3"
        assert result.params["language_code"] == "hi-IN"
        # mode should be set for saaras:v3
        assert result.params["mode"] == "translate"
        assert warnings == []

    # Removed test_transform_sarvamai_stt_with_saarika_model - model no longer in SUPPORTED_MODELS
    # The mapper logic for saarika (no mode parameter) is already tested in unit tests

    def test_transform_sarvamai_tts_with_voice(self, db: Session):
        """Test transformation of SarvamAI TTS with explicit voice."""
        kaapi_config = KaapiCompletionConfig(
            provider="sarvamai",
            type="tts",
            params={
                "model": "bulbul:v3",
                "language": "hi-IN",
                "voice": "simran",  # Explicitly set Sarvam voice to avoid cross-provider default
            },
        )

        result, warnings = transform_kaapi_config_to_native(
            session=db, kaapi_config=kaapi_config
        )

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "sarvamai-native"
        assert result.type == "tts"
        assert result.params["model"] == "bulbul:v3"
        assert result.params["target_language_code"] == "hi-IN"
        assert result.params["speaker"] == "simran"
        assert warnings == []

    def test_transform_google_text_completion(self, db: Session):
        """Text completions route through ``google-aistudio`` (AI Studio)."""
        kaapi_config = KaapiCompletionConfig(
            provider="google-aistudio",
            type="text",
            params={
                "model": "gemini-2.5-pro",
                "temperature": 0.7,
                "reasoning": "high",
            },
        )

        result, warnings = transform_kaapi_config_to_native(
            session=db, kaapi_config=kaapi_config
        )

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "google-aistudio-native"
        assert result.type == "text"
        assert result.params["model"] == "gemini-2.5-pro"
        assert result.params["temperature"] == 0.7
        assert result.params["reasoning"] == "high"
        assert warnings == []

    def test_transform_google_stt_completion(self, db: Session):
        """Test transformation of Google STT completion."""
        kaapi_config = KaapiCompletionConfig(
            provider="google",
            type="stt",
            params={"model": "gemini-2.5-pro", "instructions": "Transcribe accurately"},
        )

        result, warnings = transform_kaapi_config_to_native(
            session=db, kaapi_config=kaapi_config
        )

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "google-native"
        assert result.type == "stt"
        assert result.params["model"] == "gemini-2.5-pro"
        assert result.params["instructions"] == "Transcribe accurately"
        assert warnings == []

    def test_transform_google_tts_completion(self, db: Session):
        """Test transformation of Google TTS completion."""
        kaapi_config = KaapiCompletionConfig(
            provider="google",
            type="tts",
            params={
                "model": "gemini-2.5-flash-preview-tts",  # Updated to TTS model
                "voice": "Kore",  # Updated to supported voice
                "language": "hi-IN",  # Use BCP-47 locale that maps to Gemini lang
            },
        )

        result, warnings = transform_kaapi_config_to_native(
            session=db, kaapi_config=kaapi_config
        )

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "google-native"
        assert result.type == "tts"
        assert result.params["model"] == "gemini-2.5-flash-preview-tts"
        assert result.params["voice"] == "Kore"
        assert result.params["language"] == "hi"  # Mapped from hi-IN
        assert result.params["response_format"] == "wav"  # Default
        assert warnings == []
