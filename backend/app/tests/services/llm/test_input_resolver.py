"""
Unit tests for LLM input resolver functions.

Tests input resolution for text, base64 audio, and URL audio inputs.
"""

import base64
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from app.models.llm.request import TextInput, AudioBase64Input, AudioUrlInput
from app.services.llm.input_resolver import (
    get_file_extension,
    resolve_input,
    resolve_audio_base64,
    resolve_audio_url,
    cleanup_temp_file,
)


class TestGetFileExtension:
    """Test MIME type to file extension mapping."""

    def test_common_audio_formats(self):
        """Test common audio MIME types."""
        assert get_file_extension("audio/wav") == ".wav"
        assert get_file_extension("audio/mp3") == ".mp3"
        assert get_file_extension("audio/mpeg") == ".mp3"
        assert get_file_extension("audio/ogg") == ".ogg"

    def test_wav_variants(self):
        """Test various WAV MIME type variants."""
        assert get_file_extension("audio/wave") == ".wav"
        assert get_file_extension("audio/x-wav") == ".wav"

    def test_unknown_mime_type(self):
        """Test fallback for unknown MIME types."""
        assert get_file_extension("audio/unknown") == ".audio"
        assert get_file_extension("application/octet-stream") == ".audio"


class TestResolveInput:
    """Test main input resolution function."""

    def test_text_input(self):
        """Test resolving text input."""
        text_input = TextInput(content="Hello world")
        content, error = resolve_input(text_input)

        assert content == "Hello world"
        assert error is None

    def test_audio_base64_input(self):
        """Test resolving base64 audio input."""
        # Create minimal valid audio data
        audio_data = b"RIFF" + b"\x00" * 36  # Minimal WAV header
        encoded = base64.b64encode(audio_data).decode()

        audio_input = AudioBase64Input(data=encoded, mime_type="audio/wav")
        file_path, error = resolve_input(audio_input)

        assert error is None
        assert file_path != ""
        assert Path(file_path).exists()
        assert file_path.endswith(".wav")

        # Cleanup
        cleanup_temp_file(file_path)

    def test_audio_url_input_success(self):
        """Test resolving audio URL input."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "audio/mp3"}
        mock_response.content = b"fake audio data"

        with patch("app.services.llm.input_resolver.requests.get") as mock_get, patch(
            "app.services.llm.input_resolver.validate_callback_url"
        ):
            mock_get.return_value = mock_response

            audio_input = AudioUrlInput(url="https://example.com/audio.mp3")
            file_path, error = resolve_input(audio_input)

            assert error is None
            assert file_path != ""
            assert Path(file_path).exists()
            assert file_path.endswith(".mp3")

            # Cleanup
            cleanup_temp_file(file_path)

    def test_invalid_base64_data(self):
        """Test handling of invalid base64 data."""
        audio_input = AudioBase64Input(
            data="not-valid-base64!!!", mime_type="audio/wav"
        )
        content, error = resolve_input(audio_input)

        assert content == ""
        assert error is not None
        assert "base64" in error.lower()


class TestResolveAudioBase64:
    """Test base64 audio resolution."""

    def test_valid_base64_audio(self):
        """Test decoding valid base64 audio data."""
        audio_data = b"Test audio content"
        encoded = base64.b64encode(audio_data).decode()

        file_path, error = resolve_audio_base64(encoded, "audio/mp3")

        assert error is None
        assert file_path != ""
        assert Path(file_path).exists()
        assert file_path.endswith(".mp3")

        # Verify content
        with open(file_path, "rb") as f:
            assert f.read() == audio_data

        # Cleanup
        cleanup_temp_file(file_path)

    def test_invalid_base64_string(self):
        """Test handling invalid base64 string."""
        file_path, error = resolve_audio_base64("invalid!!!base64", "audio/wav")

        assert file_path == ""
        assert error is not None
        assert "Invalid base64" in error

    def test_different_mime_types(self):
        """Test file extension based on MIME type."""
        audio_data = b"Audio"
        encoded = base64.b64encode(audio_data).decode()

        # Test WAV
        file_path, _ = resolve_audio_base64(encoded, "audio/wav")
        assert file_path.endswith(".wav")
        cleanup_temp_file(file_path)

        # Test OGG
        file_path, _ = resolve_audio_base64(encoded, "audio/ogg")
        assert file_path.endswith(".ogg")
        cleanup_temp_file(file_path)


class TestResolveAudioUrl:
    """Test URL audio resolution with SSRF protection."""

    def test_successful_audio_fetch(self):
        """Test successfully fetching audio from URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "audio/wav"}
        mock_response.content = b"Audio file content"

        with patch("app.services.llm.input_resolver.requests.get") as mock_get, patch(
            "app.services.llm.input_resolver.validate_callback_url"
        ):
            mock_get.return_value = mock_response

            file_path, error = resolve_audio_url("https://example.com/audio.wav")

            assert error is None
            assert file_path != ""
            assert Path(file_path).exists()
            assert file_path.endswith(".wav")

            # Verify content
            with open(file_path, "rb") as f:
                assert f.read() == b"Audio file content"

            cleanup_temp_file(file_path)

    def test_ssrf_protection(self):
        """Test SSRF protection blocks dangerous URLs."""
        with patch(
            "app.services.llm.input_resolver.validate_callback_url"
        ) as mock_validate:
            mock_validate.side_effect = ValueError("SSRF attack blocked")

            file_path, error = resolve_audio_url("http://169.254.169.254/metadata")

            assert file_path == ""
            assert error is not None
            assert "Invalid audio URL" in error

    def test_http_error(self):
        """Test handling HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        with patch("app.services.llm.input_resolver.requests.get") as mock_get, patch(
            "app.services.llm.input_resolver.validate_callback_url"
        ):
            mock_get.return_value = mock_response

            file_path, error = resolve_audio_url("https://example.com/missing.wav")

            assert file_path == ""
            assert error is not None

    def test_timeout(self):
        """Test handling request timeout."""
        import requests

        with patch("app.services.llm.input_resolver.requests.get") as mock_get, patch(
            "app.services.llm.input_resolver.validate_callback_url"
        ):
            mock_get.side_effect = requests.Timeout("Request timed out")

            file_path, error = resolve_audio_url("https://example.com/slow.wav")

            assert file_path == ""
            assert error is not None
            assert "Timeout" in error

    def test_content_type_with_charset(self):
        """Test parsing content-type header with charset."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "audio/mpeg; charset=utf-8"}
        mock_response.content = b"MP3 data"

        with patch("app.services.llm.input_resolver.requests.get") as mock_get, patch(
            "app.services.llm.input_resolver.validate_callback_url"
        ):
            mock_get.return_value = mock_response

            file_path, error = resolve_audio_url("https://example.com/audio.mp3")

            assert error is None
            assert file_path.endswith(".mp3")
            cleanup_temp_file(file_path)


class TestCleanupTempFile:
    """Test temporary file cleanup."""

    def test_cleanup_existing_file(self):
        """Test cleaning up an existing temp file."""
        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            temp_path = tmp.name

        assert Path(temp_path).exists()

        # Cleanup
        cleanup_temp_file(temp_path)

        # Verify deleted
        assert not Path(temp_path).exists()

    def test_cleanup_nonexistent_file(self):
        """Test cleaning up a non-existent file (should not error)."""
        # Should not raise an exception
        cleanup_temp_file("/tmp/nonexistent_file_12345.wav")

    def test_cleanup_invalid_path(self):
        """Test cleanup with invalid path (should not error)."""
        # Should handle gracefully
        cleanup_temp_file("")
