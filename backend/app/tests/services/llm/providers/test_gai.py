"""
Tests for the Google AI provider (STT and TTS).
"""

import base64

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from app.models.llm import (
    NativeCompletionConfig,
    QueryParams,
)
from app.services.llm.providers.gai import GoogleAIProvider


def mock_google_response(
    text: str = "Transcribed text",
    model: str = "gemini-2.5-pro",
    response_id: str = "resp_123",
) -> SimpleNamespace:
    """Create a mock Google AI response object."""
    usage = SimpleNamespace(
        prompt_token_count=50,
        candidates_token_count=100,
        total_token_count=150,
        thoughts_token_count=0,
    )

    response = SimpleNamespace(
        response_id=response_id,
        model_version=model,
        text=text,
        usage_metadata=usage,
        model_dump=lambda: {
            "response_id": response_id,
            "model_version": model,
            "text": text,
            "usage_metadata": {
                "prompt_token_count": 50,
                "candidates_token_count": 100,
                "total_token_count": 150,
                "thoughts_token_count": 0,
            },
        },
    )
    return response


class TestGoogleAIProviderSTT:
    """Test cases for GoogleAIProvider STT functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Google AI client."""
        client = MagicMock()
        # Mock file upload
        mock_file = MagicMock()
        mock_file.name = "test_audio.wav"
        client.files.upload.return_value = mock_file
        return client

    @pytest.fixture
    def provider(self, mock_client):
        """Create a GoogleAIProvider instance with mock client."""
        return GoogleAIProvider(client=mock_client)

    @pytest.fixture
    def stt_config(self):
        """Create a basic STT completion config."""
        return NativeCompletionConfig(
            provider="google-native",
            type="stt",
            params={
                "model": "gemini-2.5-pro",
            },
        )

    @pytest.fixture
    def query_params(self):
        """Create basic query parameters."""
        return QueryParams(input="Test audio input")

    def test_stt_success_with_auto_language(
        self, provider, mock_client, stt_config, query_params
    ):
        """Test successful STT execution with auto language detection."""
        mock_response = mock_google_response(text="Hello world")
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(stt_config, query_params, "/path/to/audio.wav")

        assert error is None
        assert result is not None
        assert result.response.output.content.value == "Hello world"
        assert result.response.model == "gemini-2.5-pro"
        assert result.response.provider == "google-native"
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 100
        assert result.usage.total_tokens == 150

        # Verify file upload and content generation
        mock_client.files.upload.assert_called_once_with(file="/path/to/audio.wav")
        mock_client.models.generate_content.assert_called_once()

        # Verify instruction contains auto-detect
        call_args = mock_client.models.generate_content.call_args
        assert "Detect the spoken language automatically" in call_args[1]["contents"][0]

    def test_stt_with_specific_input_language(
        self, provider, mock_client, stt_config, query_params
    ):
        """Test STT with specific input language."""
        stt_config.params["input_language"] = "English"

        mock_response = mock_google_response(text="Transcribed English text")
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(stt_config, query_params, "/path/to/audio.wav")

        assert error is None
        assert result is not None

        # Verify instruction contains specific language
        call_args = mock_client.models.generate_content.call_args
        assert "Transcribe the audio from English" in call_args[1]["contents"][0]

    def test_stt_with_translation(
        self, provider, mock_client, stt_config, query_params
    ):
        """Test STT with translation to different output language."""
        stt_config.params["input_language"] = "Spanish"
        stt_config.params["output_language"] = "English"

        mock_response = mock_google_response(text="Translated text")
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(stt_config, query_params, "/path/to/audio.wav")

        assert error is None
        assert result is not None

        # Verify instruction contains translation
        call_args = mock_client.models.generate_content.call_args
        instruction = call_args[1]["contents"][0]
        assert "Transcribe the audio from Spanish" in instruction
        assert "translate to English" in instruction

    def test_stt_with_custom_instructions(
        self, provider, mock_client, stt_config, query_params
    ):
        """Test STT with custom instructions."""
        stt_config.params["instructions"] = "Include timestamps"

        mock_response = mock_google_response(text="Transcribed with timestamps")
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(stt_config, query_params, "/path/to/audio.wav")

        assert error is None
        assert result is not None

        # Verify custom instructions are included
        call_args = mock_client.models.generate_content.call_args
        instruction = call_args[1]["contents"][0]
        assert "Include timestamps" in instruction

    def test_stt_with_include_provider_raw_response(
        self, provider, mock_client, stt_config, query_params
    ):
        """Test STT with include_provider_raw_response=True."""
        mock_response = mock_google_response(text="Raw response test")
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(
            stt_config,
            query_params,
            "/path/to/audio.wav",
            include_provider_raw_response=True,
        )

        assert error is None
        assert result is not None
        assert result.provider_raw_response is not None
        assert isinstance(result.provider_raw_response, dict)
        assert result.provider_raw_response["text"] == "Raw response test"

    def test_stt_missing_model_parameter(self, provider, mock_client, query_params):
        """Test error handling when model parameter is missing."""
        stt_config = NativeCompletionConfig(
            provider="google-native",
            type="stt",
            params={},  # Missing model
        )

        result, error = provider.execute(stt_config, query_params, "/path/to/audio.wav")

        assert result is None
        assert error is not None
        assert "Missing 'model' in native params" in error

    def test_stt_with_type_error(self, provider, mock_client, stt_config, query_params):
        """Test handling of TypeError (invalid parameters)."""
        mock_client.models.generate_content.side_effect = TypeError(
            "unexpected keyword argument 'invalid_param'"
        )

        result, error = provider.execute(stt_config, query_params, "/path/to/audio.wav")

        assert result is None
        assert error is not None
        assert "Invalid or unexpected parameter in Config" in error

    def test_stt_with_generic_exception(
        self, provider, mock_client, stt_config, query_params
    ):
        """Test handling of unexpected exceptions."""
        mock_client.files.upload.side_effect = Exception("File upload failed")

        result, error = provider.execute(stt_config, query_params, "/path/to/audio.wav")

        assert result is None
        assert error is not None
        assert "Unexpected error occurred" in error

    def test_stt_with_invalid_input_type(
        self, provider, mock_client, stt_config, query_params
    ):
        """Test STT execution with invalid input type (non-string)."""
        # Pass a dict instead of a string path
        invalid_input = {"invalid": "data"}

        result, error = provider.execute(stt_config, query_params, invalid_input)

        assert result is None
        assert error is not None
        assert "STT requires file path as string" in error

    def test_stt_with_valid_file_path(
        self, provider, mock_client, stt_config, query_params
    ):
        """Test STT execution with valid file path string."""
        mock_response = mock_google_response(text="Valid transcription")
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(stt_config, query_params, "/path/to/audio.wav")

        assert error is None
        assert result is not None
        assert result.response.output.content.value == "Valid transcription"


def mock_tts_google_response(
    audio_bytes: bytes = b"\x00\x01\x02\x03",
    model: str = "gemini-2.5-pro-preview-tts",
    response_id: str = "resp_tts_123",
) -> SimpleNamespace:
    """Create a mock Google AI TTS response object with audio data."""
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=0,
        total_token_count=10,
        thoughts_token_count=0,
    )

    inline_data = SimpleNamespace(data=audio_bytes)
    part = SimpleNamespace(inline_data=inline_data)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)

    response = SimpleNamespace(
        response_id=response_id,
        model_version=model,
        candidates=[candidate],
        usage_metadata=usage,
        model_dump=lambda: {
            "response_id": response_id,
            "model_version": model,
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 0,
                "total_token_count": 10,
                "thoughts_token_count": 0,
            },
        },
    )
    return response


SAMPLE_PCM_BYTES = b"\x00\x01" * 1000


class TestGoogleAIProviderTTS:
    """Test cases for GoogleAIProvider TTS functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Google AI client."""
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_client):
        """Create a GoogleAIProvider instance with mock client."""
        return GoogleAIProvider(client=mock_client)

    @pytest.fixture
    def tts_config(self):
        """Create a basic TTS completion config."""
        return NativeCompletionConfig(
            provider="google-native",
            type="tts",
            params={
                "model": "gemini-2.5-pro-preview-tts",
                "voice": "Kore",
                "language": "en-US",
            },
        )

    @pytest.fixture
    def query_params(self):
        """Create basic query parameters."""
        return QueryParams(input="Hello world")

    def test_tts_success_wav_default(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test successful TTS execution with default WAV format."""
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello world")

        assert error is None
        assert result is not None
        assert result.response.output.type == "audio"
        assert result.response.output.content.format == "base64"
        assert result.response.output.content.mime_type == "audio/wav"
        # Verify the base64 content decodes back to original bytes
        decoded = base64.b64decode(result.response.output.content.value)
        assert decoded == SAMPLE_PCM_BYTES
        assert result.response.provider_response_id == "resp_tts_123"
        assert result.response.model == "gemini-2.5-pro-preview-tts"
        assert result.response.provider == "google-native"
        assert result.usage.input_tokens == 10
        assert result.usage.total_tokens == 10

    def test_tts_success_mp3_format(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS with MP3 response format conversion."""
        tts_config.params["response_format"] = "mp3"
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        fake_mp3_bytes = b"fake-mp3-content"
        with patch(
            "app.services.llm.providers.gai.convert_pcm_to_mp3",
            return_value=(fake_mp3_bytes, None),
        ) as mock_convert:
            result, error = provider.execute(tts_config, query_params, "Hello world")

        assert error is None
        assert result is not None
        assert result.response.output.content.mime_type == "audio/mp3"
        decoded = base64.b64decode(result.response.output.content.value)
        assert decoded == fake_mp3_bytes
        mock_convert.assert_called_once_with(SAMPLE_PCM_BYTES)

    def test_tts_success_ogg_format(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS with OGG response format conversion."""
        tts_config.params["response_format"] = "ogg"
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        fake_ogg_bytes = b"fake-ogg-content"
        with patch(
            "app.services.llm.providers.gai.convert_pcm_to_ogg",
            return_value=(fake_ogg_bytes, None),
        ) as mock_convert:
            result, error = provider.execute(tts_config, query_params, "Hello world")

        assert error is None
        assert result is not None
        assert result.response.output.content.mime_type == "audio/ogg"
        decoded = base64.b64decode(result.response.output.content.value)
        assert decoded == fake_ogg_bytes
        mock_convert.assert_called_once_with(SAMPLE_PCM_BYTES)

    def test_tts_mp3_conversion_failure(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test error when MP3 conversion fails."""
        tts_config.params["response_format"] = "mp3"
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "app.services.llm.providers.gai.convert_pcm_to_mp3",
            return_value=(None, "ffmpeg not found"),
        ):
            result, error = provider.execute(tts_config, query_params, "Hello world")

        assert result is None
        assert "Failed to convert audio to MP3" in error
        assert "ffmpeg not found" in error

    def test_tts_ogg_conversion_failure(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test error when OGG conversion fails."""
        tts_config.params["response_format"] = "ogg"
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "app.services.llm.providers.gai.convert_pcm_to_ogg",
            return_value=(None, "codec error"),
        ):
            result, error = provider.execute(tts_config, query_params, "Hello world")

        assert result is None
        assert "Failed to convert audio to OGG" in error

    def test_tts_empty_input(self, provider, mock_client, tts_config, query_params):
        """Test error when text input is empty."""
        result, error = provider.execute(tts_config, query_params, "   ")

        assert result is None
        assert error == "Text input cannot be empty"
        mock_client.models.generate_content.assert_not_called()

    def test_tts_non_string_input(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test error when input is not a string."""
        result, error = provider.execute(tts_config, query_params, {"invalid": "data"})

        assert result is None
        assert "TTS requires text string as input" in error

    def test_tts_missing_model(self, provider, mock_client, query_params):
        """Test error when model parameter is missing."""
        config = NativeCompletionConfig(
            provider="google-native",
            type="tts",
            params={"voice": "Kore", "language": "en-US"},
        )

        result, error = provider.execute(config, query_params, "Hello")

        assert result is None
        assert error == "Missing 'model' in native params"

    def test_tts_missing_voice(self, provider, mock_client, query_params):
        """Test error when voice parameter is missing."""
        config = NativeCompletionConfig(
            provider="google-native",
            type="tts",
            params={"model": "gemini-2.5-pro-preview-tts", "language": "en-US"},
        )

        result, error = provider.execute(config, query_params, "Hello")

        assert result is None
        assert error == "Missing 'voice' in native params"

    def test_tts_missing_language(self, provider, mock_client, query_params):
        """Test error when language parameter is missing."""
        config = NativeCompletionConfig(
            provider="google-native",
            type="tts",
            params={"model": "gemini-2.5-pro-preview-tts", "voice": "Kore"},
        )

        result, error = provider.execute(config, query_params, "Hello")

        assert result is None
        assert error == "Missing 'language' in native params"

    def test_tts_missing_response_id(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test error when response has no response_id."""
        mock_response = mock_tts_google_response()
        mock_response.response_id = None
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert error == "Google AI response missing response_id"

    def test_tts_missing_audio_data(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test error when response has no audio data in parts."""
        mock_response = mock_tts_google_response(audio_bytes=None)
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert error == "Google AI response missing audio data"

    def test_tts_empty_candidates(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test error when response has empty candidates list."""
        mock_response = mock_tts_google_response()
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert "Failed to extract audio from response" in error

    def test_tts_missing_inline_data(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test error when candidate part has no inline_data."""
        mock_response = mock_tts_google_response()
        mock_response.candidates[0].content.parts[0].inline_data = None
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert "Failed to extract audio from response" in error

    def test_tts_no_usage_metadata(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS response when usage_metadata is None (defaults to zeros)."""
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        assert result is not None
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.usage.total_tokens == 0
        assert result.usage.reasoning_tokens == 0

    def test_tts_include_provider_raw_response(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS with include_provider_raw_response=True."""
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(
            tts_config, query_params, "Hello", include_provider_raw_response=True
        )

        assert error is None
        assert result is not None
        assert result.provider_raw_response is not None
        assert isinstance(result.provider_raw_response, dict)

    def test_tts_without_provider_raw_response(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS without raw response (default)."""
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        assert result.provider_raw_response is None

    def test_tts_with_director_notes(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS with Gemini-specific director_notes parameter."""
        tts_config.params["provider_specific"] = {
            "gemini": {"director_notes": "Speak in a cheerful tone"}
        }
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        assert result is not None
        # Verify config was passed with system_instruction
        call_args = mock_client.models.generate_content.call_args
        config_arg = call_args[1]["config"]
        assert config_arg.system_instruction == "Speak in a cheerful tone"

    def test_tts_without_director_notes(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS without director_notes (no system_instruction in config)."""
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        call_args = mock_client.models.generate_content.call_args
        config_arg = call_args[1]["config"]
        assert (
            not hasattr(config_arg, "system_instruction")
            or config_arg.system_instruction is None
        )

    def test_tts_model_version_fallback(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test that model falls back to config model when model_version is None."""
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_response.model_version = None
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        assert result.response.model == "gemini-2.5-pro-preview-tts"

    def test_tts_generic_exception(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test handling of unexpected exceptions."""
        mock_client.models.generate_content.side_effect = Exception("API unavailable")

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert error == "Unexpected error occurred"

    def test_tts_type_error(self, provider, mock_client, tts_config, query_params):
        """Test handling of TypeError (invalid parameters)."""
        mock_client.models.generate_content.side_effect = TypeError(
            "unexpected keyword argument"
        )

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert result is None
        assert "Invalid or unexpected parameter in Config" in error

    def test_tts_passes_correct_model_and_content(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test that the correct model and text content are passed to the API."""
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Say this text")

        assert error is None
        call_args = mock_client.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-2.5-pro-preview-tts"
        assert call_args[1]["contents"] == "Say this text"

    def test_tts_passes_correct_voice_config(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test that voice and language are correctly configured."""
        mock_response = mock_tts_google_response(audio_bytes=SAMPLE_PCM_BYTES)
        mock_client.models.generate_content.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        call_args = mock_client.models.generate_content.call_args
        config_arg = call_args[1]["config"]
        assert config_arg.response_modalities == ["AUDIO"]
        voice_name = (
            config_arg.speech_config.voice_config.prebuilt_voice_config.voice_name
        )
        assert voice_name == "Kore"
        assert config_arg.speech_config.language_code == "en-US"
