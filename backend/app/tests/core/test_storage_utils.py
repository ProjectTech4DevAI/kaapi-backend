"""Tests for storage_utils.py - upload and load functions for object store."""

import json
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from app.core.cloud.storage import CloudStorageError
from app.core.storage_utils import (
    load_json_from_object_store,
    upload_jsonl_to_object_store,
)


class TestUploadToObjectStore:
    """Test uploading content to object store."""

    # ==================== Upload Success Tests ====================

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

    # ==================== Upload Failure Tests ====================

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


class TestLoadJsonFromObjectStore:
    """Test loading JSON from object store."""

    # ==================== Load Success Tests ====================

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

    # ==================== Load Failure Tests ====================

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
