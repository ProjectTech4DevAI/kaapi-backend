"""Test cases for Gemini client wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.stt_evaluations.gemini.client import (
    GeminiClient,
    GeminiClientError,
)


class TestGeminiClientInit:
    """Test cases for GeminiClient initialization."""

    @patch("app.services.stt_evaluations.gemini.client.genai.Client")
    def test_initialization_with_api_key(self, mock_genai_client):
        """Test client initialization with API key."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        client = GeminiClient(api_key="test-api-key")

        mock_genai_client.assert_called_once_with(api_key="test-api-key")
        assert client._api_key == "test-api-key"

    @patch("app.services.stt_evaluations.gemini.client.genai.Client")
    def test_client_property(self, mock_genai_client):
        """Test client property returns underlying client."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        client = GeminiClient(api_key="test-api-key")

        assert client.client == mock_client_instance


class TestGeminiClientFromCredentials:
    """Test cases for GeminiClient.from_credentials class method."""

    @patch("app.services.stt_evaluations.gemini.client.genai.Client")
    @patch("app.services.stt_evaluations.gemini.client.get_provider_credential")
    def test_successful_creation(self, mock_get_creds, mock_genai_client):
        """Test successful client creation from credentials."""
        mock_get_creds.return_value = {"api_key": "stored-api-key"}
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        mock_session = MagicMock()

        client = GeminiClient.from_credentials(
            session=mock_session,
            org_id=1,
            project_id=2,
        )

        mock_get_creds.assert_called_once_with(
            session=mock_session,
            org_id=1,
            project_id=2,
            provider="google",
        )
        assert client._api_key == "stored-api-key"

    @patch("app.services.stt_evaluations.gemini.client.get_provider_credential")
    def test_credentials_not_found(self, mock_get_creds):
        """Test error when credentials are not found."""
        from app.core.exception_handlers import HTTPException

        mock_get_creds.return_value = None
        mock_session = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            GeminiClient.from_credentials(
                session=mock_session,
                org_id=1,
                project_id=2,
            )

        assert exc_info.value.status_code == 404
        assert "credentials not configured" in str(exc_info.value.detail)

    @patch("app.services.stt_evaluations.gemini.client.get_provider_credential")
    def test_missing_api_key(self, mock_get_creds):
        """Test error when credentials exist but api_key is missing."""
        mock_get_creds.return_value = {"other_field": "value"}  # No api_key
        mock_session = MagicMock()

        with pytest.raises(GeminiClientError) as exc_info:
            GeminiClient.from_credentials(
                session=mock_session,
                org_id=1,
                project_id=2,
            )

        assert "missing api_key" in str(exc_info.value)

    @patch("app.services.stt_evaluations.gemini.client.get_provider_credential")
    def test_empty_api_key(self, mock_get_creds):
        """Test error when api_key is empty."""
        mock_get_creds.return_value = {"api_key": ""}  # Empty api_key
        mock_session = MagicMock()

        with pytest.raises(GeminiClientError) as exc_info:
            GeminiClient.from_credentials(
                session=mock_session,
                org_id=1,
                project_id=2,
            )

        assert "missing api_key" in str(exc_info.value)


class TestGeminiClientError:
    """Test cases for GeminiClientError exception."""

    def test_error_message(self):
        """Test error message is preserved."""
        error = GeminiClientError("Test error message")
        assert str(error) == "Test error message"

    def test_error_inheritance(self):
        """Test error inherits from Exception."""
        error = GeminiClientError("Test")
        assert isinstance(error, Exception)
