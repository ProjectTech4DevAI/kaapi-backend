"""
Tests for the SarvamAI provider (STT and TTS).
"""

import base64
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from types import SimpleNamespace

from app.models.llm import (
    NativeCompletionConfig,
    QueryParams,
)
from app.services.llm.providers.sai import SarvamAIProvider


def mock_sarvam_stt_response(
    transcript: str = "नमस्ते",
    request_id: str = "req_stt_123",
) -> SimpleNamespace:
    """Create a mock SarvamAI STT response object."""
    response = SimpleNamespace(
        transcript=transcript,
        request_id=request_id,
        model_dump=lambda: {
            "transcript": transcript,
            "request_id": request_id,
        },
    )
    return response


def mock_sarvam_tts_response(
    audio_base64: str = "YXVkaW9kYXRh",
    request_id: str = "req_tts_456",
) -> SimpleNamespace:
    """Create a mock SarvamAI TTS response object."""
    response = SimpleNamespace(
        audios=[audio_base64],
        request_id=request_id,
        model_dump=lambda: {
            "audios": [audio_base64],
            "request_id": request_id,
        },
    )
    return response


class TestSarvamAIProviderSTT:
    """Test cases for SarvamAIProvider STT functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock SarvamAI client."""
        client = MagicMock()
        client.speech_to_text = MagicMock()
        return client

    @pytest.fixture
    def provider(self, mock_client):
        """Create a SarvamAIProvider instance with mock client."""
        return SarvamAIProvider(client=mock_client)

    @pytest.fixture
    def stt_config(self):
        """Create a basic STT completion config."""
        return NativeCompletionConfig(
            provider="sarvamai-native",
            type="stt",
            params={
                "model": "saarika:v1",
                "language_code": "hi-IN",
                "mode": "transcribe",
            },
        )

    @pytest.fixture
    def query_params(self):
        """Create basic query parameters."""
        return QueryParams(input="Test audio input")

    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create a temporary audio file for testing."""
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_bytes(b"fake audio data")
        return str(audio_file)

    def test_stt_success_basic_transcription(
        self, provider, mock_client, stt_config, query_params, temp_audio_file
    ):
        """Test successful STT transcription."""
        mock_response = mock_sarvam_stt_response(transcript="नमस्ते दुनिया")
        mock_client.speech_to_text.transcribe.return_value = mock_response

        result, error = provider.execute(stt_config, query_params, temp_audio_file)

        assert error is None
        assert result is not None
        assert result.response.output.content.value == "नमस्ते दुनिया"
        assert result.response.model == "saarika:v1"
        assert result.response.provider == "sarvamai-native"
        assert result.response.provider_response_id == "req_stt_123"
        assert result.usage.output_tokens == 2  # Number of words

    def test_stt_success_with_translate_mode(
        self, provider, mock_client, query_params, temp_audio_file
    ):
        """Test STT with translate mode."""
        config = NativeCompletionConfig(
            provider="sarvamai-native",
            type="stt",
            params={
                "model": "saarika:v1",
                "language_code": "hi-IN",
                "mode": "translate",
            },
        )
        mock_response = mock_sarvam_stt_response(transcript="Hello world")
        mock_client.speech_to_text.transcribe.return_value = mock_response

        result, error = provider.execute(config, query_params, temp_audio_file)

        assert error is None
        assert result is not None
        assert result.response.output.content.value == "Hello world"
        # Verify translate mode was passed to API
        call_args = mock_client.speech_to_text.transcribe.call_args
        assert call_args.kwargs["mode"] == "translate"

    def test_stt_success_with_unknown_language(
        self, provider, mock_client, query_params, temp_audio_file
    ):
        """Test STT with unknown/auto language detection."""
        config = NativeCompletionConfig(
            provider="sarvamai-native",
            type="stt",
            params={
                "model": "saarika:v1",
                "language_code": "unknown",
                "mode": "transcribe",
            },
        )
        mock_response = mock_sarvam_stt_response(transcript="Detected text")
        mock_client.speech_to_text.transcribe.return_value = mock_response

        result, error = provider.execute(config, query_params, temp_audio_file)

        assert error is None
        assert result is not None
        call_args = mock_client.speech_to_text.transcribe.call_args
        assert call_args.kwargs["language_code"] == "unknown"

    def test_stt_missing_model_param(
        self, provider, mock_client, query_params, temp_audio_file
    ):
        """Test STT with missing model parameter."""
        config = NativeCompletionConfig(
            provider="sarvamai-native",
            type="stt",
            params={
                "language_code": "hi-IN",
                "mode": "transcribe",
            },
        )

        result, error = provider.execute(config, query_params, temp_audio_file)

        assert result is None
        assert error is not None
        assert "model" in error.lower()

    def test_stt_invalid_file_path(
        self, provider, mock_client, stt_config, query_params
    ):
        """Test STT with non-existent file path."""
        result, error = provider.execute(
            stt_config, query_params, "/nonexistent/path/audio.wav"
        )

        assert result is None
        assert error is not None

    def test_stt_api_exception(
        self, provider, mock_client, stt_config, query_params, temp_audio_file
    ):
        """Test STT when API raises exception."""
        mock_client.speech_to_text.transcribe.side_effect = Exception(
            "API connection failed"
        )

        result, error = provider.execute(stt_config, query_params, temp_audio_file)

        assert result is None
        assert error is not None
        assert "API connection failed" in error

    def test_stt_include_provider_raw_response(
        self, provider, mock_client, stt_config, query_params, temp_audio_file
    ):
        """Test STT with include_provider_raw_response flag."""
        mock_response = mock_sarvam_stt_response(transcript="Test")
        mock_client.speech_to_text.transcribe.return_value = mock_response

        result, error = provider.execute(
            stt_config,
            query_params,
            temp_audio_file,
            include_provider_raw_response=True,
        )

        assert error is None
        assert result is not None
        assert result.provider_raw_response is not None
        assert result.provider_raw_response["transcript"] == "Test"
        assert result.provider_raw_response["request_id"] == "req_stt_123"


class TestSarvamAIProviderTTS:
    """Test cases for SarvamAIProvider TTS functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock SarvamAI client."""
        client = MagicMock()
        client.text_to_speech = MagicMock()
        return client

    @pytest.fixture
    def provider(self, mock_client):
        """Create a SarvamAIProvider instance with mock client."""
        return SarvamAIProvider(client=mock_client)

    @pytest.fixture
    def tts_config(self):
        """Create a basic TTS completion config."""
        return NativeCompletionConfig(
            provider="sarvamai-native",
            type="tts",
            params={
                "model": "bulbul:v1",
                "target_language_code": "hi-IN",
                "speaker": "meera",
                "output_audio_codec": "wav",
            },
        )

    @pytest.fixture
    def query_params(self):
        """Create basic query parameters."""
        return QueryParams(input="Test text input")

    def test_tts_success_basic_conversion(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test successful TTS conversion."""
        audio_data = base64.b64encode(b"fake audio binary data").decode("utf-8")
        mock_response = mock_sarvam_tts_response(audio_base64=audio_data)
        mock_client.text_to_speech.convert.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "नमस्ते दुनिया")

        assert error is None
        assert result is not None
        assert result.response.output.content.value == audio_data
        assert result.response.output.content.format == "base64"
        assert result.response.output.content.mime_type == "audio/wav"
        assert result.response.model == "bulbul:v1"
        assert result.response.provider == "sarvamai-native"

    def test_tts_with_mp3_codec(self, provider, mock_client, query_params):
        """Test TTS with MP3 codec."""
        config = NativeCompletionConfig(
            provider="sarvamai-native",
            type="tts",
            params={
                "model": "bulbul:v1",
                "target_language_code": "en-IN",
                "speaker": "arvind",
                "output_audio_codec": "mp3",
            },
        )
        audio_data = base64.b64encode(b"mp3 audio data").decode("utf-8")
        mock_response = mock_sarvam_tts_response(audio_base64=audio_data)
        mock_client.text_to_speech.convert.return_value = mock_response

        result, error = provider.execute(config, query_params, "Hello world")

        assert error is None
        assert result is not None
        assert result.response.output.content.mime_type == "audio/mp3"
        call_args = mock_client.text_to_speech.convert.call_args
        assert call_args.kwargs["output_audio_codec"] == "mp3"

    def test_tts_with_ogg_codec(self, provider, mock_client, query_params):
        """Test TTS with OGG codec."""
        config = NativeCompletionConfig(
            provider="sarvamai-native",
            type="tts",
            params={
                "model": "bulbul:v1",
                "target_language_code": "hi-IN",
                "speaker": "meera",
                "output_audio_codec": "ogg",
            },
        )
        audio_data = base64.b64encode(b"ogg audio data").decode("utf-8")
        mock_response = mock_sarvam_tts_response(audio_base64=audio_data)
        mock_client.text_to_speech.convert.return_value = mock_response

        result, error = provider.execute(config, query_params, "Test text")

        assert error is None
        assert result is not None
        assert result.response.output.content.mime_type == "audio/ogg"

    def test_tts_missing_model_param(self, provider, mock_client, query_params):
        """Test TTS with missing model parameter."""
        config = NativeCompletionConfig(
            provider="sarvamai-native",
            type="tts",
            params={
                "target_language_code": "hi-IN",
                "speaker": "meera",
            },
        )

        result, error = provider.execute(config, query_params, "Test text")

        assert result is None
        assert error is not None
        assert "model" in error.lower()

    def test_tts_missing_target_language_code(
        self, provider, mock_client, query_params
    ):
        """Test TTS with missing target_language_code."""
        config = NativeCompletionConfig(
            provider="sarvamai-native",
            type="tts",
            params={
                "model": "bulbul:v1",
                "speaker": "meera",
            },
        )

        result, error = provider.execute(config, query_params, "Test text")

        assert result is None
        assert error is not None
        assert "target_language_code" in error.lower()

    def test_tts_empty_audio_response(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS when API returns empty audio list."""
        mock_response = SimpleNamespace(
            audios=[],
            request_id="req_123",
            model_dump=lambda: {"audios": [], "request_id": "req_123"},
        )
        mock_client.text_to_speech.convert.return_value = mock_response

        result, error = provider.execute(tts_config, query_params, "Test text")

        assert result is None
        assert error is not None
        assert "no audio data" in error.lower()

    def test_tts_api_exception(self, provider, mock_client, tts_config, query_params):
        """Test TTS when API raises exception."""
        mock_client.text_to_speech.convert.side_effect = Exception(
            "TTS service unavailable"
        )

        result, error = provider.execute(tts_config, query_params, "Test text")

        assert result is None
        assert error is not None
        assert "TTS service unavailable" in error

    def test_tts_include_provider_raw_response(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS with include_provider_raw_response flag."""
        audio_data = base64.b64encode(b"audio data").decode("utf-8")
        mock_response = mock_sarvam_tts_response(audio_base64=audio_data)
        mock_client.text_to_speech.convert.return_value = mock_response

        result, error = provider.execute(
            tts_config,
            query_params,
            "Test text",
            include_provider_raw_response=True,
        )

        assert error is None
        assert result is not None
        assert result.provider_raw_response is not None
        assert result.provider_raw_response["audios"] == [audio_data]

    def test_tts_usage_estimates(self, provider, mock_client, tts_config, query_params):
        """Test that TTS properly estimates token usage based on input text."""
        audio_data = base64.b64encode(b"audio").decode("utf-8")
        mock_response = mock_sarvam_tts_response(audio_base64=audio_data)
        mock_client.text_to_speech.convert.return_value = mock_response

        # Test with multi-word input
        result, error = provider.execute(
            tts_config, query_params, "Hello world how are you"
        )

        assert error is None
        assert result.usage.input_tokens == 5  # 5 words
        assert result.usage.output_tokens == 0  # Audio has no output tokens
        assert result.usage.total_tokens == 5


class TestSarvamAIProviderClientCreation:
    """Test cases for SarvamAIProvider client creation."""

    def test_create_client_with_valid_api_key(self):
        """Test client creation with valid API key."""
        credentials = {"api_key": "test_api_key_123"}

        with patch("app.services.llm.providers.sai.SarvamAI") as mock_sarvam_class:
            client = SarvamAIProvider.create_client(credentials)

            mock_sarvam_class.assert_called_once_with(
                api_subscription_key="test_api_key_123"
            )

    def test_create_client_missing_api_key(self):
        """Test client creation with missing API key."""
        credentials = {}

        with pytest.raises(ValueError) as exc_info:
            SarvamAIProvider.create_client(credentials)

        assert "API Key for SarvamAI Not Set" in str(exc_info.value)

    def test_create_client_empty_credentials(self):
        """Test client creation with empty credentials dict."""
        credentials = {"other_key": "value"}

        with pytest.raises(ValueError) as exc_info:
            SarvamAIProvider.create_client(credentials)

        assert "API Key for SarvamAI Not Set" in str(exc_info.value)


class TestSarvamAIProviderInputParsing:
    """Test cases for SarvamAIProvider input parsing."""

    @pytest.fixture
    def provider(self):
        """Create a SarvamAIProvider instance."""
        mock_client = MagicMock()
        return SarvamAIProvider(client=mock_client)

    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create a temporary audio file."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"audio data")
        return str(audio_file)

    def test_parse_input_stt_valid_file(self, provider, temp_audio_file):
        """Test parsing valid file path for STT."""
        result = provider._parse_input(temp_audio_file, "stt", "sarvamai")
        assert result == temp_audio_file

    def test_parse_input_stt_invalid_file(self, provider):
        """Test parsing invalid file path for STT."""
        with pytest.raises(ValueError) as exc_info:
            provider._parse_input("/nonexistent/file.wav", "stt", "sarvamai")

        assert "valid file path" in str(exc_info.value)

    def test_parse_input_tts_valid_text(self, provider):
        """Test parsing valid text for TTS."""
        result = provider._parse_input("Hello world", "tts", "sarvamai")
        assert result == "Hello world"

    def test_parse_input_tts_invalid_type(self, provider):
        """Test parsing invalid type for TTS."""
        with pytest.raises(ValueError) as exc_info:
            provider._parse_input(12345, "tts", "sarvamai")

        assert "text string" in str(exc_info.value)

    def test_parse_input_unsupported_completion_type(self, provider):
        """Test parsing with unsupported completion type."""
        with pytest.raises(ValueError) as exc_info:
            provider._parse_input("input", "unsupported", "sarvamai")

        assert "Unsupported completion type" in str(exc_info.value)


class TestSarvamAIProviderExecute:
    """Test cases for SarvamAIProvider execute method."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock SarvamAI client."""
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_client):
        """Create a SarvamAIProvider instance."""
        return SarvamAIProvider(client=mock_client)

    @pytest.fixture
    def query_params(self):
        """Create basic query parameters."""
        return QueryParams(input="Test input")

    def test_execute_unsupported_completion_type(self, provider, query_params):
        """Test execute with unsupported completion type."""
        config = NativeCompletionConfig(
            provider="sarvamai-native",
            type="text",  # Unsupported for SarvamAI
            params={"model": "test-model"},
        )

        result, error = provider.execute(config, query_params, "input")

        assert result is None
        assert error is not None
        assert "Unsupported completion type" in error
