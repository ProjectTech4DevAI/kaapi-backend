"""Test cases for STT audio upload service."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile

from app.core.exception_handlers import HTTPException
from app.services.stt_evaluations.audio import (
    _resolve_extension,
    _validate_audio_file,
    upload_audio_file,
)
from app.services.stt_evaluations.constants import MAX_FILE_SIZE_BYTES


def _make_upload_file(
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


class TestResolveExtension:
    """Test cases for _resolve_extension."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("audio.mp3", "mp3"),
            ("audio.wav", "wav"),
            ("audio.flac", "flac"),
            ("audio.m4a", "m4a"),
            ("audio.ogg", "ogg"),
            ("audio.webm", "webm"),
            ("audio.MP3", "mp3"),
            ("audio.Mp3", "mp3"),
            ("audio.backup.mp3", "mp3"),
            (".audio.mp3", "mp3"),
            ("/path/to/audio.mp3", "mp3"),
        ],
    )
    def test_extension_from_filename(self, filename: str, expected: str) -> None:
        file = _make_upload_file(filename=filename)
        assert _resolve_extension(file) == expected

    @pytest.mark.parametrize(
        "filename",
        ["", None, "audiofile"],
    )
    def test_no_extension_from_filename(self, filename: str | None) -> None:
        file = _make_upload_file(filename=filename, content_type=None)
        assert _resolve_extension(file) is None

    @pytest.mark.parametrize(
        "content_type, expected",
        [
            ("audio/mpeg", "mp3"),
            ("audio/mp3", "mp3"),
            ("audio/wav", "wav"),
            ("audio/x-wav", "wav"),
            ("audio/wave", "wav"),
            ("audio/flac", "flac"),
            ("audio/mp4", "m4a"),
            ("audio/ogg", "ogg"),
            ("audio/webm", "webm"),
            ("AUDIO/MPEG", "mp3"),
        ],
    )
    def test_fallback_to_content_type(self, content_type: str, expected: str) -> None:
        file = _make_upload_file(filename="audiofile", content_type=content_type)
        assert _resolve_extension(file) == expected

    def test_unknown_content_type(self) -> None:
        file = _make_upload_file(
            filename="audiofile", content_type="application/octet-stream"
        )
        assert _resolve_extension(file) is None


class TestValidateAudioFile:
    """Test cases for _validate_audio_file."""

    @pytest.mark.parametrize("ext", ["mp3", "wav", "flac", "m4a", "ogg", "webm"])
    def test_valid_formats(self, ext: str) -> None:
        file = _make_upload_file(filename=f"test.{ext}")
        assert _validate_audio_file(file) == ext

    def test_missing_filename(self) -> None:
        file = _make_upload_file(filename=None)
        with pytest.raises(HTTPException) as exc_info:
            _validate_audio_file(file)
        assert exc_info.value.status_code == 400
        assert "Filename is required" in str(exc_info.value.detail)

    def test_empty_filename(self) -> None:
        file = _make_upload_file(filename="")
        with pytest.raises(HTTPException) as exc_info:
            _validate_audio_file(file)
        assert exc_info.value.status_code == 400

    def test_unsupported_format(self) -> None:
        file = _make_upload_file(filename="test.txt")
        with pytest.raises(HTTPException) as exc_info:
            _validate_audio_file(file)
        assert exc_info.value.status_code == 400
        assert "Unsupported audio format" in str(exc_info.value.detail)

    def test_content_type_fallback(self) -> None:
        file = _make_upload_file(filename="audiofile", content_type="audio/mpeg")
        assert _validate_audio_file(file) == "mp3"

    def test_file_too_large(self) -> None:
        file = _make_upload_file(size=MAX_FILE_SIZE_BYTES + 1)
        with pytest.raises(HTTPException) as exc_info:
            _validate_audio_file(file)
        assert exc_info.value.status_code == 400
        assert "File too large" in str(exc_info.value.detail)

    def test_file_at_max_size(self) -> None:
        file = _make_upload_file(size=MAX_FILE_SIZE_BYTES)
        assert _validate_audio_file(file) == "mp3"

    def test_file_with_no_size(self) -> None:
        file = _make_upload_file(size=None)
        assert _validate_audio_file(file) == "mp3"


class TestUploadAudioFile:
    """Test cases for upload_audio_file."""

    @patch("app.services.stt_evaluations.audio.create_file")
    @patch("app.services.stt_evaluations.audio.get_cloud_storage")
    def test_successful_upload(self, mock_get_storage, mock_create_file) -> None:
        mock_storage = MagicMock()
        mock_storage.put.return_value = "s3://bucket/stt/audio/test.mp3"
        mock_storage.get_file_size_kb.return_value = 1.0
        mock_get_storage.return_value = mock_storage

        mock_file_record = MagicMock()
        mock_file_record.id = 1
        mock_create_file.return_value = mock_file_record

        result = upload_audio_file(
            session=MagicMock(),
            file=_make_upload_file(),
            organization_id=1,
            project_id=1,
        )

        assert result.file_id == 1
        assert result.s3_url == "s3://bucket/stt/audio/test.mp3"
        assert result.filename == "test.mp3"
        assert result.size_bytes == 1024
        assert result.content_type == "audio/mpeg"

    @patch("app.services.stt_evaluations.audio.get_cloud_storage")
    def test_upload_validation_error(self, mock_get_storage) -> None:
        with pytest.raises(HTTPException) as exc_info:
            upload_audio_file(
                session=MagicMock(),
                file=_make_upload_file(filename="test.txt"),
                organization_id=1,
                project_id=1,
            )
        assert exc_info.value.status_code == 400
        assert "Unsupported audio format" in str(exc_info.value.detail)

    @patch("app.services.stt_evaluations.audio.get_cloud_storage")
    def test_upload_storage_error(self, mock_get_storage) -> None:
        mock_storage = MagicMock()
        mock_storage.put.side_effect = Exception("S3 connection failed")
        mock_get_storage.return_value = mock_storage

        with pytest.raises(HTTPException) as exc_info:
            upload_audio_file(
                session=MagicMock(),
                file=_make_upload_file(),
                organization_id=1,
                project_id=1,
            )
        assert exc_info.value.status_code == 500
        assert "Failed to upload audio file" in str(exc_info.value.detail)

    @patch("app.services.stt_evaluations.audio.create_file")
    @patch("app.services.stt_evaluations.audio.get_cloud_storage")
    def test_upload_uses_file_size_on_s3_error(
        self, mock_get_storage, mock_create_file
    ) -> None:
        mock_storage = MagicMock()
        mock_storage.put.return_value = "s3://bucket/stt/audio/test.mp3"
        mock_storage.get_file_size_kb.side_effect = Exception("Failed to get size")
        mock_get_storage.return_value = mock_storage

        mock_file_record = MagicMock()
        mock_file_record.id = 1
        mock_create_file.return_value = mock_file_record

        result = upload_audio_file(
            session=MagicMock(),
            file=_make_upload_file(size=2048),
            organization_id=1,
            project_id=1,
        )
        assert result.size_bytes == 2048
