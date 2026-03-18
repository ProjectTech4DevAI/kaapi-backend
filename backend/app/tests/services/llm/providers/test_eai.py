"""
Tests for the ElevenLabs provider (STT and TTS).
"""

import base64
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from app.models.llm import (
    NativeCompletionConfig,
    QueryParams,
)
from app.services.llm.providers.eai import ElevenlabsAIProvider


def mock_elevenlabs_stt_response(
    text: str = "Hello world",
    language_code: str = "eng",
    transcription_id: str = "txn_stt_123",
) -> SimpleNamespace:
    """Create a mock ElevenLabs STT response object."""
    response = SimpleNamespace(
        text=text,
        language_code=language_code,
        language_probability=0.98,
        transcription_id=transcription_id,
        words=[],
        model_dump=lambda: {
            "text": text,
            "language_code": language_code,
            "language_probability": 0.98,
            "transcription_id": transcription_id,
            "words": [],
        },
    )
    return response


class TestElevenlabsProviderSTT:
    """Test cases for ElevenlabsAIProvider STT functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock ElevenLabs client."""
        client = MagicMock()
        client.speech_to_text = MagicMock()
        return client

    @pytest.fixture
    def provider(self, mock_client):
        """Create an ElevenlabsAIProvider instance with mock client."""
        return ElevenlabsAIProvider(client=mock_client)

    @pytest.fixture
    def stt_config(self):
        """Create a basic STT completion config."""
        return NativeCompletionConfig(
            provider="elevenlabs-native",
            type="stt",
            params={
                "model_id": "scribe_v1",
                "language_code": "hin",
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
        """Test successful STT transcription with Hindi audio."""
        mock_response = mock_elevenlabs_stt_response(
            text="namaste duniya", language_code="hin"
        )
        mock_client.speech_to_text.convert.return_value = mock_response

        result, error = provider.execute(stt_config, query_params, temp_audio_file)

        assert error is None
        assert result is not None
        assert result.response.output.content.value == "namaste duniya"
        assert result.response.model == "scribe_v1"
        assert result.response.provider == "elevenlabs-native"
        assert result.response.provider_response_id == "txn_stt_123"
        assert result.usage.output_tokens == 2

    def test_stt_auto_detect_language(
        self, provider, mock_client, query_params, temp_audio_file
    ):
        """Test STT without language_code lets ElevenLabs auto-detect."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="stt",
            params={"model_id": "scribe_v1"},
        )
        mock_response = mock_elevenlabs_stt_response(text="Detected text")
        mock_client.speech_to_text.convert.return_value = mock_response

        result, error = provider.execute(config, query_params, temp_audio_file)

        assert error is None
        assert result is not None
        call_kwargs = mock_client.speech_to_text.convert.call_args.kwargs
        assert "language_code" not in call_kwargs

    def test_stt_with_temperature(
        self, provider, mock_client, query_params, temp_audio_file
    ):
        """Test STT passes temperature to the API."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="stt",
            params={
                "model_id": "scribe_v1",
                "language_code": "eng",
                "temperature": 0.5,
            },
        )
        mock_response = mock_elevenlabs_stt_response(text="Hello")
        mock_client.speech_to_text.convert.return_value = mock_response

        result, error = provider.execute(config, query_params, temp_audio_file)

        assert error is None
        call_kwargs = mock_client.speech_to_text.convert.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_stt_uses_default_model_when_missing(
        self, provider, mock_client, query_params, temp_audio_file
    ):
        """Test STT uses default model (scribe_v2) when model_id is not provided."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="stt",
            params={"language_code": "eng"},
        )
        mock_response = mock_elevenlabs_stt_response(text="Default model test")
        mock_client.speech_to_text.convert.return_value = mock_response

        result, error = provider.execute(config, query_params, temp_audio_file)

        assert error is None
        assert result is not None
        # Verify the default model was used
        call_kwargs = mock_client.speech_to_text.convert.call_args.kwargs
        assert call_kwargs["model_id"] == "scribe_v2"

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
        """Test STT when ElevenLabs API raises an exception."""
        mock_client.speech_to_text.convert.side_effect = Exception(
            "API rate limit exceeded"
        )

        result, error = provider.execute(stt_config, query_params, temp_audio_file)

        assert result is None
        assert error is not None
        assert "API rate limit exceeded" in error

    def test_stt_include_provider_raw_response(
        self, provider, mock_client, stt_config, query_params, temp_audio_file
    ):
        """Test STT with include_provider_raw_response flag."""
        mock_response = mock_elevenlabs_stt_response(text="Test transcript")
        mock_client.speech_to_text.convert.return_value = mock_response

        result, error = provider.execute(
            stt_config,
            query_params,
            temp_audio_file,
            include_provider_raw_response=True,
        )

        assert error is None
        assert result is not None
        assert result.provider_raw_response is not None
        assert result.provider_raw_response["text"] == "Test transcript"
        assert result.provider_raw_response["transcription_id"] == "txn_stt_123"


class TestElevenlabsProviderTTS:
    """Test cases for ElevenlabsAIProvider TTS functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock ElevenLabs client."""
        client = MagicMock()
        client.text_to_speech = MagicMock()
        return client

    @pytest.fixture
    def provider(self, mock_client):
        """Create an ElevenlabsAIProvider instance with mock client."""
        return ElevenlabsAIProvider(client=mock_client)

    @pytest.fixture
    def tts_config(self):
        """Create a basic TTS completion config."""
        return NativeCompletionConfig(
            provider="elevenlabs-native",
            type="tts",
            params={
                "model_id": "eleven_v3",
                "voice_id": "JBFqnCBsd6RMkjVDRZzb",
                "language_code": "hin",
                "output_format": "mp3_44100_128",
            },
        )

    @pytest.fixture
    def query_params(self):
        """Create basic query parameters."""
        return QueryParams(input="Test text input")

    def test_tts_success_basic_conversion(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test successful TTS conversion returns base64 audio."""
        audio_bytes = b"fake mp3 audio binary data"
        mock_client.text_to_speech.convert.return_value = iter([audio_bytes])

        result, error = provider.execute(tts_config, query_params, "Namaste duniya")

        assert error is None
        assert result is not None
        expected_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        assert result.response.output.content.value == expected_b64
        assert result.response.output.content.format == "base64"
        assert result.response.output.content.mime_type == "audio/mpeg"
        assert result.response.model == "eleven_v3"

        assert result.response.provider == "elevenlabs-native"

    def test_tts_chunked_audio_response(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS correctly joins chunked/streamed audio bytes."""
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        mock_client.text_to_speech.convert.return_value = iter(chunks)

        result, error = provider.execute(tts_config, query_params, "Hello")

        assert error is None
        expected_b64 = base64.b64encode(b"chunk1chunk2chunk3").decode("utf-8")
        assert result.response.output.content.value == expected_b64

    def test_tts_wav_output_format(self, provider, mock_client, query_params):
        """Test TTS with WAV output format sets correct mime type."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="tts",
            params={
                "model_id": "eleven_v3",
                "voice_id": "JBFqnCBsd6RMkjVDRZzb",
                "output_format": "wav_24000",
            },
        )
        mock_client.text_to_speech.convert.return_value = iter([b"wav data"])

        result, error = provider.execute(config, query_params, "Test")

        assert error is None
        assert result.response.output.content.mime_type == "audio/wav"
        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["output_format"] == "wav_24000"

    def test_tts_opus_output_format(self, provider, mock_client, query_params):
        """Test TTS with Opus output format sets correct mime type."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="tts",
            params={
                "model_id": "eleven_v3",
                "voice_id": "JBFqnCBsd6RMkjVDRZzb",
                "output_format": "opus_48000_128",
            },
        )
        mock_client.text_to_speech.convert.return_value = iter([b"opus data"])

        result, error = provider.execute(config, query_params, "Test")

        assert error is None
        assert result.response.output.content.mime_type == "audio/opus"

    def test_tts_default_output_format(self, provider, mock_client, query_params):
        """Test TTS defaults to wav_24000 when output_format is not specified."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="tts",
            params={
                "model_id": "eleven_v3",
                "voice_id": "JBFqnCBsd6RMkjVDRZzb",
            },
        )
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        result, error = provider.execute(config, query_params, "Test")

        assert error is None
        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["output_format"] == "wav_24000"
        assert (
            result.response.output.content.mime_type == "audio/wav"
        )  # Fixed: wav format → audio/wav mime type

    def test_tts_passes_language_code(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS passes language_code as optional kwarg to SDK."""
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        provider.execute(tts_config, query_params, "Test")

        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["language_code"] == "hin"

    def test_tts_omits_language_code_when_absent(
        self, provider, mock_client, query_params
    ):
        """Test TTS does not pass language_code when not in params."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="tts",
            params={
                "model_id": "eleven_v3",
                "voice_id": "JBFqnCBsd6RMkjVDRZzb",
            },
        )
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        provider.execute(config, query_params, "Test")

        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert "language_code" not in call_kwargs

    def test_tts_uses_default_model_when_missing(
        self, provider, mock_client, query_params
    ):
        """Test TTS uses default model (eleven_turbo_v2) when model_id is not provided."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="tts",
            params={"voice_id": "JBFqnCBsd6RMkjVDRZzb"},
        )
        mock_client.text_to_speech.convert.return_value = iter([b"audio data"])

        result, error = provider.execute(config, query_params, "Test text")

        assert error is None
        assert result is not None
        # Verify the default model was used
        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["model_id"] == "eleven_v3"

    def test_tts_uses_default_voice_when_missing(
        self, provider, mock_client, query_params
    ):
        """Test TTS uses default voice (Sarah) when voice_id is not provided."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="tts",
            params={"model_id": "eleven_v3"},
        )
        mock_client.text_to_speech.convert.return_value = iter([b"audio data"])

        result, error = provider.execute(config, query_params, "Test text")

        assert error is None
        assert result is not None
        # Verify the default voice (Sarah) was used
        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["voice_id"] == "EXAVITQu4vr4xnSDxMaL"  # Sarah's ID

    def test_tts_empty_audio_response(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS when API returns empty audio iterator."""
        mock_client.text_to_speech.convert.return_value = iter([])

        result, error = provider.execute(tts_config, query_params, "Test text")

        assert result is None
        assert error is not None
        assert "no audio data" in error.lower()

    def test_tts_api_exception(self, provider, mock_client, tts_config, query_params):
        """Test TTS when ElevenLabs API raises an exception."""
        mock_client.text_to_speech.convert.side_effect = Exception("TTS quota exceeded")

        result, error = provider.execute(tts_config, query_params, "Test text")

        assert result is None
        assert error is not None
        assert "TTS quota exceeded" in error

    def test_tts_include_provider_raw_response(
        self, provider, mock_client, tts_config, query_params
    ):
        """Test TTS with include_provider_raw_response flag."""
        audio_bytes = b"audio data for raw response test"
        mock_client.text_to_speech.convert.return_value = iter([audio_bytes])

        result, error = provider.execute(
            tts_config,
            query_params,
            "Test text",
            include_provider_raw_response=True,
        )

        assert error is None
        assert result.provider_raw_response is not None
        assert result.provider_raw_response["audio_bytes_length"] == len(audio_bytes)
        assert result.provider_raw_response["output_format"] == "mp3_44100_128"

    def test_tts_usage_estimates(self, provider, mock_client, tts_config, query_params):
        """Test that TTS properly estimates token usage based on input text."""
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        result, error = provider.execute(
            tts_config, query_params, "Hello world how are you"
        )

        assert error is None
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 0
        assert result.usage.total_tokens == 5


class TestElevenlabsProviderClientCreation:
    """Test cases for ElevenlabsAIProvider client creation."""

    def test_create_client_with_valid_api_key(self):
        """Test client creation with valid API key."""
        credentials = {"api_key": "test_api_key_123"}

        with patch(
            "app.services.llm.providers.eai.ElevenLabs"
        ) as mock_elevenlabs_class:
            client = ElevenlabsAIProvider.create_client(credentials)

            mock_elevenlabs_class.assert_called_once_with(api_key="test_api_key_123")

    def test_create_client_missing_api_key(self):
        """Test client creation with missing API key."""
        credentials = {}

        with pytest.raises(ValueError) as exc_info:
            ElevenlabsAIProvider.create_client(credentials)

        assert "API Key for Elevenlabs Not Set" in str(exc_info.value)

    def test_create_client_wrong_credential_key(self):
        """Test client creation with wrong credential key name."""
        credentials = {"secret_key": "value"}

        with pytest.raises(ValueError) as exc_info:
            ElevenlabsAIProvider.create_client(credentials)

        assert "API Key for Elevenlabs Not Set" in str(exc_info.value)


class TestElevenlabsProviderExecute:
    """Test cases for ElevenlabsAIProvider execute routing."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock ElevenLabs client."""
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_client):
        """Create an ElevenlabsAIProvider instance."""
        return ElevenlabsAIProvider(client=mock_client)

    @pytest.fixture
    def query_params(self):
        """Create basic query parameters."""
        return QueryParams(input="Test input")

    def test_execute_unsupported_completion_type(self, provider, query_params):
        """Test execute with unsupported completion type returns error."""
        config = NativeCompletionConfig(
            provider="elevenlabs-native",
            type="text",
            params={"model_id": "test-model"},
        )

        result, error = provider.execute(config, query_params, "input")

        assert result is None
        assert error is not None
        assert "Unsupported completion type" in error
