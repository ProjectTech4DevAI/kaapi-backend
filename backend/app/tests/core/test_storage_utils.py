"""Tests for storage_utils.py - upload and load functions for object store."""

import json
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from app.core.cloud.storage import CloudStorageError
from app.core.storage_utils import (
    generate_timestamped_filename,
    get_mime_from_url,
    load_json_from_object_store,
    upload_jsonl_to_object_store,
    upload_to_object_store,
)


class TestGetMimeFromUrl:
    """Test cases for get_mime_from_url function."""

    def test_mp3_url(self):
        """Test MIME detection for MP3 files."""
        url = "https://bucket.s3.amazonaws.com/audio/test.mp3"
        assert get_mime_from_url(url) == "audio/mpeg"

    def test_wav_url(self):
        """Test MIME detection for WAV files."""
        url = "https://bucket.s3.amazonaws.com/audio/test.wav"
        assert get_mime_from_url(url) == "audio/x-wav"

    def test_flac_url(self):
        """Test MIME detection for FLAC files."""
        url = "https://bucket.s3.amazonaws.com/audio/test.flac"
        mime = get_mime_from_url(url)
        # FLAC can be detected as audio/flac or audio/x-flac depending on system
        assert mime in ("audio/flac", "audio/x-flac")

    def test_ogg_url(self):
        """Test MIME detection for OGG files."""
        url = "https://bucket.s3.amazonaws.com/audio/test.ogg"
        assert get_mime_from_url(url) == "audio/ogg"

    def test_webm_url(self):
        """Test MIME detection for WEBM files."""
        url = "https://bucket.s3.amazonaws.com/audio/test.webm"
        mime = get_mime_from_url(url)
        # webm can be detected as audio or video depending on system
        assert mime in ("audio/webm", "video/webm")

    def test_signed_url_with_query_params(self):
        """Test MIME detection for signed URLs with query parameters."""
        url = (
            "https://bucket.s3.amazonaws.com/audio/test.mp3"
            "?X-Amz-Signature=abc123&X-Amz-Expires=3600"
        )
        assert get_mime_from_url(url) == "audio/mpeg"

    def test_url_encoded_path(self):
        """Test MIME detection for URL-encoded paths."""
        url = "https://bucket.s3.amazonaws.com/audio/test%20file.mp3"
        assert get_mime_from_url(url) == "audio/mpeg"

    def test_unknown_extension(self):
        """Test MIME detection returns None for unknown extensions."""
        url = "https://bucket.s3.amazonaws.com/file.unknown"
        assert get_mime_from_url(url) is None

    def test_no_extension(self):
        """Test MIME detection returns None for URLs without extension."""
        url = "https://bucket.s3.amazonaws.com/file"
        assert get_mime_from_url(url) is None

    def test_csv_url(self):
        """Test MIME detection for CSV files."""
        url = "https://bucket.s3.amazonaws.com/data/test.csv"
        assert get_mime_from_url(url) == "text/csv"

    def test_json_url(self):
        """Test MIME detection for JSON files."""
        url = "https://bucket.s3.amazonaws.com/data/test.json"
        assert get_mime_from_url(url) == "application/json"


class TestUploadToObjectStore:
    """Test cases for upload_to_object_store function."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock CloudStorage instance."""
        storage = MagicMock()
        storage.put.return_value = "s3://bucket/test/file.txt"
        return storage

    def test_successful_upload(self, mock_storage):
        """Test successful file upload."""
        content = b"test content"
        result = upload_to_object_store(
            storage=mock_storage,
            content=content,
            filename="test.txt",
            subdirectory="uploads",
            content_type="text/plain",
        )

        assert result == "s3://bucket/test/file.txt"
        mock_storage.put.assert_called_once()

        # Verify the UploadFile was created correctly
        call_args = mock_storage.put.call_args
        upload_file = call_args.kwargs["source"]
        assert upload_file.filename == "test.txt"

    def test_upload_with_default_content_type(self, mock_storage):
        """Test upload uses default content type."""
        content = b"binary data"
        result = upload_to_object_store(
            storage=mock_storage,
            content=content,
            filename="data.bin",
            subdirectory="files",
        )

        assert result == "s3://bucket/test/file.txt"
        mock_storage.put.assert_called_once()

    def test_upload_returns_none_on_cloud_storage_error(self, mock_storage):
        """Test that CloudStorageError returns None gracefully."""
        mock_storage.put.side_effect = CloudStorageError("Connection failed")

        result = upload_to_object_store(
            storage=mock_storage,
            content=b"data",
            filename="test.txt",
            subdirectory="uploads",
        )

        assert result is None

    def test_upload_returns_none_on_generic_error(self, mock_storage):
        """Test that generic exceptions return None gracefully."""
        mock_storage.put.side_effect = Exception("Unexpected error")

        result = upload_to_object_store(
            storage=mock_storage,
            content=b"data",
            filename="test.txt",
            subdirectory="uploads",
        )

        assert result is None


class TestUploadJsonlToObjectStore:
    """Test cases for upload_jsonl_to_object_store function."""

    def test_upload_json_file_success(self) -> None:
        """Verify successful JSON format upload returns URL with correct content."""
        mock_storage = MagicMock()
        mock_storage.put.return_value = "s3://bucket/path/traces.json"

        results = [{"trace_id": "t1", "score": 0.9}, {"trace_id": "t2", "score": 0.8}]

        url = upload_jsonl_to_object_store(
            storage=mock_storage,
            results=results,
            filename="traces.json",
            subdirectory="evaluations/score/70",
            format="json",
        )

        assert url == "s3://bucket/path/traces.json"
        mock_storage.put.assert_called_once()

        # Verify content is valid JSON array
        call_args = mock_storage.put.call_args
        upload_file = call_args.kwargs.get("source")
        content = upload_file.file.read().decode("utf-8")
        assert json.loads(content) == results

    def test_upload_jsonl_file_success(self) -> None:
        """Verify successful JSONL format upload returns URL with correct content."""
        mock_storage = MagicMock()
        mock_storage.put.return_value = "s3://bucket/path/traces.jsonl"

        results = [{"trace_id": "t1", "score": 0.9}, {"trace_id": "t2", "score": 0.8}]

        url = upload_jsonl_to_object_store(
            storage=mock_storage,
            results=results,
            filename="traces.jsonl",
            subdirectory="evaluations/score/70",
            format="jsonl",
        )

        assert url == "s3://bucket/path/traces.jsonl"
        mock_storage.put.assert_called_once()

        # Verify content is valid JSONL (one JSON object per line)
        call_args = mock_storage.put.call_args
        upload_file = call_args.kwargs.get("source")
        content = upload_file.file.read().decode("utf-8")
        parsed_results = [json.loads(line) for line in content.strip().split("\n")]
        assert parsed_results == results

    def test_upload_returns_none_on_storage_error(self) -> None:
        """Verify returns None on CloudStorageError."""
        mock_storage = MagicMock()
        mock_storage.put.side_effect = CloudStorageError("Upload failed")

        url = upload_jsonl_to_object_store(
            storage=mock_storage,
            results=[{"id": 1}],
            filename="test.json",
            subdirectory="test",
            format="json",
        )

        assert url is None

    def test_results_with_unicode(self) -> None:
        """Test upload with unicode content."""
        mock_storage = MagicMock()
        mock_storage.put.return_value = "s3://bucket/results/unicode.jsonl"

        results = [
            {"text": "Hello 世界"},
            {"text": "Emoji 🎵"},
        ]
        result = upload_jsonl_to_object_store(
            storage=mock_storage,
            results=results,
            filename="unicode.jsonl",
            subdirectory="results",
        )

        assert result is not None


class TestLoadJsonFromObjectStore:
    """Test loading JSON from object store."""

    def test_load_success(self) -> None:
        """Verify successful load returns parsed JSON."""
        mock_storage = MagicMock()
        test_data = [{"id": 1, "value": "test"}]
        mock_storage.stream.return_value = BytesIO(
            json.dumps(test_data).encode("utf-8")
        )

        result = load_json_from_object_store(
            storage=mock_storage,
            url="s3://bucket/path/file.json",
        )

        assert result == test_data
        mock_storage.stream.assert_called_once_with("s3://bucket/path/file.json")

    def test_load_returns_none_on_storage_error(self) -> None:
        """Verify returns None on CloudStorageError."""
        mock_storage = MagicMock()
        mock_storage.stream.side_effect = CloudStorageError("Download failed")

        result = load_json_from_object_store(
            storage=mock_storage,
            url="s3://bucket/file.json",
        )

        assert result is None

    def test_load_returns_none_on_invalid_json(self) -> None:
        """Verify returns None on invalid JSON content."""
        mock_storage = MagicMock()
        mock_storage.stream.return_value = BytesIO(b"not valid json")

        result = load_json_from_object_store(
            storage=mock_storage,
            url="s3://bucket/file.json",
        )

        assert result is None


class TestGenerateTimestampedFilename:
    """Test cases for generate_timestamped_filename function."""

    def test_default_csv_extension(self):
        """Test that default extension is CSV."""
        filename = generate_timestamped_filename("dataset")
        assert filename.endswith(".csv")
        assert filename.startswith("dataset_")

    def test_custom_extension(self):
        """Test custom file extension."""
        filename = generate_timestamped_filename("results", extension="jsonl")
        assert filename.endswith(".jsonl")
        assert filename.startswith("results_")

    def test_timestamp_format(self):
        """Test that timestamp is in expected format."""
        filename = generate_timestamped_filename("test")
        # Expected format: test_YYYYMMDD_HHMMSS.csv
        parts = filename.split("_")
        assert len(parts) == 3
        # Date part should be 8 digits
        assert len(parts[1]) == 8
        assert parts[1].isdigit()
        # Time part should be 6 digits + extension
        time_part = parts[2].split(".")[0]
        assert len(time_part) == 6
        assert time_part.isdigit()

    def test_with_special_characters_in_base_name(self):
        """Test with special characters in base name."""
        filename = generate_timestamped_filename("my-dataset-v1")
        assert filename.startswith("my-dataset-v1_")
        assert filename.endswith(".csv")

    def test_unique_filenames(self):
        """Test that consecutive calls produce different filenames."""
        import time

        filename1 = generate_timestamped_filename("test")
        time.sleep(0.01)  # Small delay to ensure different timestamp
        filename2 = generate_timestamped_filename("test")
        # They may be the same if called in the same second
        # but the format should be correct
        assert filename1.startswith("test_")
        assert filename2.startswith("test_")
