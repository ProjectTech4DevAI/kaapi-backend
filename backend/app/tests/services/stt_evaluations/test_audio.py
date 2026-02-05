"""Test cases for STT audio validation and upload service."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile

from app.services.stt_evaluations.audio import (
    AudioValidationError,
    get_extension_from_content_type,
    get_extension_from_filename,
    upload_audio_file,
    validate_audio_file,
    validate_s3_audio_url,
)
from app.services.stt_evaluations.constants import MAX_FILE_SIZE_BYTES


class TestGetExtensionFromFilename:
    """Test cases for get_extension_from_filename function."""

    def test_mp3_extension(self):
        """Test extracting MP3 extension."""
        assert get_extension_from_filename("audio.mp3") == "mp3"

    def test_wav_extension(self):
        """Test extracting WAV extension."""
        assert get_extension_from_filename("audio.wav") == "wav"

    def test_flac_extension(self):
        """Test extracting FLAC extension."""
        assert get_extension_from_filename("audio.flac") == "flac"

    def test_m4a_extension(self):
        """Test extracting M4A extension."""
        assert get_extension_from_filename("audio.m4a") == "m4a"

    def test_ogg_extension(self):
        """Test extracting OGG extension."""
        assert get_extension_from_filename("audio.ogg") == "ogg"

    def test_webm_extension(self):
        """Test extracting WEBM extension."""
        assert get_extension_from_filename("audio.webm") == "webm"

    def test_uppercase_extension(self):
        """Test that uppercase extensions are normalized to lowercase."""
        assert get_extension_from_filename("audio.MP3") == "mp3"
        assert get_extension_from_filename("audio.WAV") == "wav"

    def test_mixed_case_extension(self):
        """Test mixed case extensions."""
        assert get_extension_from_filename("audio.Mp3") == "mp3"

    def test_empty_filename(self):
        """Test empty filename returns None."""
        assert get_extension_from_filename("") is None

    def test_none_filename(self):
        """Test None filename returns None."""
        assert get_extension_from_filename(None) is None

    def test_no_extension(self):
        """Test filename without extension returns None."""
        assert get_extension_from_filename("audiofile") is None

    def test_multiple_dots(self):
        """Test filename with multiple dots."""
        assert get_extension_from_filename("audio.backup.mp3") == "mp3"

    def test_hidden_file_with_extension(self):
        """Test hidden file with extension."""
        assert get_extension_from_filename(".audio.mp3") == "mp3"

    def test_path_with_filename(self):
        """Test full path with filename."""
        assert get_extension_from_filename("/path/to/audio.mp3") == "mp3"


class TestGetExtensionFromContentType:
    """Test cases for get_extension_from_content_type function."""

    def test_audio_mpeg(self):
        """Test audio/mpeg content type."""
        assert get_extension_from_content_type("audio/mpeg") == "mp3"

    def test_audio_mp3(self):
        """Test audio/mp3 content type."""
        assert get_extension_from_content_type("audio/mp3") == "mp3"

    def test_audio_wav(self):
        """Test audio/wav content type."""
        assert get_extension_from_content_type("audio/wav") == "wav"

    def test_audio_x_wav(self):
        """Test audio/x-wav content type."""
        assert get_extension_from_content_type("audio/x-wav") == "wav"

    def test_audio_wave(self):
        """Test audio/wave content type."""
        assert get_extension_from_content_type("audio/wave") == "wav"

    def test_audio_flac(self):
        """Test audio/flac content type."""
        assert get_extension_from_content_type("audio/flac") == "flac"

    def test_audio_mp4(self):
        """Test audio/mp4 content type (m4a)."""
        assert get_extension_from_content_type("audio/mp4") == "m4a"

    def test_audio_ogg(self):
        """Test audio/ogg content type."""
        assert get_extension_from_content_type("audio/ogg") == "ogg"

    def test_audio_webm(self):
        """Test audio/webm content type."""
        assert get_extension_from_content_type("audio/webm") == "webm"

    def test_uppercase_content_type(self):
        """Test uppercase content type is normalized."""
        assert get_extension_from_content_type("AUDIO/MPEG") == "mp3"

    def test_empty_content_type(self):
        """Test empty content type returns None."""
        assert get_extension_from_content_type("") is None

    def test_none_content_type(self):
        """Test None content type returns None."""
        assert get_extension_from_content_type(None) is None

    def test_unknown_content_type(self):
        """Test unknown content type returns None."""
        assert get_extension_from_content_type("application/octet-stream") is None


class TestValidateAudioFile:
    """Test cases for validate_audio_file function."""

    def _create_upload_file(
        self,
        filename: str | None = "test.mp3",
        content_type: str | None = "audio/mpeg",
        size: int | None = 1024,
    ) -> UploadFile:
        """Create a mock UploadFile for testing."""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = filename
        mock_file.content_type = content_type
        mock_file.size = size
        return mock_file

    def test_valid_mp3_file(self):
        """Test validation of valid MP3 file."""
        file = self._create_upload_file(filename="test.mp3")
        result = validate_audio_file(file)
        assert result == "mp3"

    def test_valid_wav_file(self):
        """Test validation of valid WAV file."""
        file = self._create_upload_file(filename="test.wav")
        result = validate_audio_file(file)
        assert result == "wav"

    def test_valid_flac_file(self):
        """Test validation of valid FLAC file."""
        file = self._create_upload_file(filename="test.flac")
        result = validate_audio_file(file)
        assert result == "flac"

    def test_valid_m4a_file(self):
        """Test validation of valid M4A file."""
        file = self._create_upload_file(filename="test.m4a")
        result = validate_audio_file(file)
        assert result == "m4a"

    def test_valid_ogg_file(self):
        """Test validation of valid OGG file."""
        file = self._create_upload_file(filename="test.ogg")
        result = validate_audio_file(file)
        assert result == "ogg"

    def test_valid_webm_file(self):
        """Test validation of valid WEBM file."""
        file = self._create_upload_file(filename="test.webm")
        result = validate_audio_file(file)
        assert result == "webm"

    def test_missing_filename(self):
        """Test validation fails when filename is missing."""
        file = self._create_upload_file(filename=None)
        with pytest.raises(AudioValidationError) as exc_info:
            validate_audio_file(file)
        assert "Filename is required" in str(exc_info.value)

    def test_empty_filename(self):
        """Test validation fails when filename is empty."""
        file = self._create_upload_file(filename="")
        with pytest.raises(AudioValidationError) as exc_info:
            validate_audio_file(file)
        assert "Filename is required" in str(exc_info.value)

    def test_unsupported_format(self):
        """Test validation fails for unsupported format."""
        file = self._create_upload_file(filename="test.txt")
        with pytest.raises(AudioValidationError) as exc_info:
            validate_audio_file(file)
        assert "Unsupported audio format" in str(exc_info.value)

    def test_extension_from_content_type_fallback(self):
        """Test fallback to content type when filename has no extension."""
        file = self._create_upload_file(filename="audiofile", content_type="audio/mpeg")
        result = validate_audio_file(file)
        assert result == "mp3"

    def test_file_too_large(self):
        """Test validation fails when file is too large."""
        file = self._create_upload_file(
            filename="test.mp3",
            size=MAX_FILE_SIZE_BYTES + 1,
        )
        with pytest.raises(AudioValidationError) as exc_info:
            validate_audio_file(file)
        assert "File too large" in str(exc_info.value)

    def test_file_at_max_size(self):
        """Test validation passes when file is exactly at max size."""
        file = self._create_upload_file(
            filename="test.mp3",
            size=MAX_FILE_SIZE_BYTES,
        )
        result = validate_audio_file(file)
        assert result == "mp3"

    def test_file_with_no_size(self):
        """Test validation passes when file size is not available."""
        file = self._create_upload_file(filename="test.mp3", size=None)
        result = validate_audio_file(file)
        assert result == "mp3"


class TestValidateS3AudioUrl:
    """Test cases for validate_s3_audio_url function."""

    def test_valid_s3_mp3_url(self):
        """Test valid S3 MP3 URL."""
        url = "s3://bucket/audio/test.mp3"
        assert validate_s3_audio_url(url) is True

    def test_valid_s3_wav_url(self):
        """Test valid S3 WAV URL."""
        url = "s3://bucket/audio/test.wav"
        assert validate_s3_audio_url(url) is True

    def test_valid_s3_flac_url(self):
        """Test valid S3 FLAC URL."""
        url = "s3://bucket/audio/test.flac"
        assert validate_s3_audio_url(url) is True

    def test_empty_url(self):
        """Test empty URL returns False."""
        assert validate_s3_audio_url("") is False

    def test_none_url(self):
        """Test None URL returns False."""
        assert validate_s3_audio_url(None) is False

    def test_https_url(self):
        """Test HTTPS URL returns False."""
        url = "https://bucket.s3.amazonaws.com/audio/test.mp3"
        assert validate_s3_audio_url(url) is False

    def test_s3_url_with_invalid_extension(self):
        """Test S3 URL with invalid extension returns False."""
        url = "s3://bucket/file.txt"
        assert validate_s3_audio_url(url) is False

    def test_s3_url_with_no_extension(self):
        """Test S3 URL with no extension returns False."""
        url = "s3://bucket/audiofile"
        assert validate_s3_audio_url(url) is False


class TestUploadAudioFile:
    """Test cases for upload_audio_file function."""

    def _create_upload_file(
        self,
        filename: str = "test.mp3",
        content_type: str = "audio/mpeg",
        size: int = 1024,
    ) -> UploadFile:
        """Create a mock UploadFile for testing."""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = filename
        mock_file.content_type = content_type
        mock_file.size = size
        return mock_file

    @patch("app.services.stt_evaluations.audio.create_file")
    @patch("app.services.stt_evaluations.audio.get_cloud_storage")
    def test_successful_upload(self, mock_get_storage, mock_create_file):
        """Test successful audio file upload."""
        mock_storage = MagicMock()
        mock_storage.put.return_value = "s3://bucket/stt/audio/test.mp3"
        mock_storage.get_file_size_kb.return_value = 1.0
        mock_get_storage.return_value = mock_storage

        # Mock the file record creation
        mock_file_record = MagicMock()
        mock_file_record.id = 1
        mock_create_file.return_value = mock_file_record

        mock_session = MagicMock()
        file = self._create_upload_file()

        result = upload_audio_file(
            session=mock_session,
            file=file,
            organization_id=1,
            project_id=1,
        )

        assert result.file_id == 1
        assert result.s3_url == "s3://bucket/stt/audio/test.mp3"
        assert result.filename == "test.mp3"
        assert result.size_bytes == 1024
        assert result.content_type == "audio/mpeg"

    @patch("app.services.stt_evaluations.audio.get_cloud_storage")
    def test_upload_validation_error(self, mock_get_storage):
        """Test upload fails on validation error."""
        from app.core.exception_handlers import HTTPException

        mock_session = MagicMock()
        file = self._create_upload_file(filename="test.txt")

        with pytest.raises(HTTPException) as exc_info:
            upload_audio_file(
                session=mock_session,
                file=file,
                organization_id=1,
                project_id=1,
            )

        assert exc_info.value.status_code == 400
        assert "Unsupported audio format" in str(exc_info.value.detail)

    @patch("app.services.stt_evaluations.audio.get_cloud_storage")
    def test_upload_storage_error(self, mock_get_storage):
        """Test upload handles storage errors."""
        from app.core.exception_handlers import HTTPException

        mock_storage = MagicMock()
        mock_storage.put.side_effect = Exception("S3 connection failed")
        mock_get_storage.return_value = mock_storage

        mock_session = MagicMock()
        file = self._create_upload_file()

        with pytest.raises(HTTPException) as exc_info:
            upload_audio_file(
                session=mock_session,
                file=file,
                organization_id=1,
                project_id=1,
            )

        assert exc_info.value.status_code == 500
        assert "Failed to upload audio file" in str(exc_info.value.detail)

    @patch("app.services.stt_evaluations.audio.create_file")
    @patch("app.services.stt_evaluations.audio.get_cloud_storage")
    def test_upload_uses_file_size_on_s3_error(
        self, mock_get_storage, mock_create_file
    ):
        """Test upload uses file.size when S3 size retrieval fails."""
        mock_storage = MagicMock()
        mock_storage.put.return_value = "s3://bucket/stt/audio/test.mp3"
        mock_storage.get_file_size_kb.side_effect = Exception("Failed to get size")
        mock_get_storage.return_value = mock_storage

        # Mock the file record creation
        mock_file_record = MagicMock()
        mock_file_record.id = 1
        mock_create_file.return_value = mock_file_record

        mock_session = MagicMock()
        file = self._create_upload_file(size=2048)

        result = upload_audio_file(
            session=mock_session,
            file=file,
            organization_id=1,
            project_id=1,
        )

        assert result.size_bytes == 2048
