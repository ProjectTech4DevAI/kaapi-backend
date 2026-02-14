"""
Tests for the Google AI provider (STT).
"""

import pytest
from unittest.mock import MagicMock
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
