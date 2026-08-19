"""Test cases for VertexBatchProvider (Vertex AI batch, GCS-backed)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.batch.vertex import VertexBatchProvider, _parse_gs_uri

_BUCKET = "test-bucket"


@pytest.fixture
def mock_genai():
    return MagicMock()


@pytest.fixture
def mock_storage():
    return MagicMock()


@pytest.fixture
def provider(mock_genai, mock_storage):
    return VertexBatchProvider(
        client=mock_genai,
        storage_client=mock_storage,
        gcs_bucket=_BUCKET,
        model="gemini-2.5-pro",
    )


class TestParseGsUri:
    def test_parses_bucket_and_key(self):
        assert _parse_gs_uri("gs://b/dir/f.jsonl") == ("b", "dir/f.jsonl")

    def test_rejects_non_gs(self):
        with pytest.raises(ValueError, match="gs://"):
            _parse_gs_uri("https://b/f.jsonl")


class TestCreateBatch:
    def test_uploads_to_gcs_and_starts_job(self, provider, mock_genai, mock_storage):
        job = MagicMock()
        job.name = "batches/123"
        job.state.name = "JOB_STATE_PENDING"
        mock_genai.batches.create.return_value = job

        result = provider.create_batch(
            [{"request": {"contents": []}}], {"display_name": "run-1"}
        )

        # Input JSONL uploaded to gs://bucket/batch-input/...
        blob = mock_storage.bucket.return_value.blob.return_value
        blob.upload_from_string.assert_called_once()
        # batches.create called with a gs:// src and a gs:// dest in config.
        kwargs = mock_genai.batches.create.call_args.kwargs
        assert kwargs["src"].startswith(f"gs://{_BUCKET}/batch-input/")
        assert kwargs["config"].dest.startswith(f"gs://{_BUCKET}/batch-output/")
        assert result["provider_batch_id"] == "batches/123"
        assert result["provider_status"] == "JOB_STATE_PENDING"
        assert result["total_items"] == 1


class TestGetBatchStatus:
    def test_succeeded_returns_output_uri(self, provider, mock_genai):
        job = MagicMock()
        job.state.name = "JOB_STATE_SUCCEEDED"
        job.dest.gcs_uri = "gs://b/out/"
        mock_genai.batches.get.return_value = job

        result = provider.get_batch_status("batches/123")
        assert result["provider_status"] == "JOB_STATE_SUCCEEDED"
        assert result["provider_output_file_id"] == "gs://b/out/"
        assert "error_message" not in result

    def test_failed_sets_error_message(self, provider, mock_genai):
        job = MagicMock()
        job.state.name = "JOB_STATE_FAILED"
        job.dest = None
        job.error.message = "boom"
        mock_genai.batches.get.return_value = job

        result = provider.get_batch_status("batches/123")
        assert result["provider_status"] == "JOB_STATE_FAILED"
        assert result["error_message"] == "boom"
        assert result["provider_output_file_id"] == "batches/123"


class TestDownloadBatchResults:
    def _blob(self, name, text):
        blob = MagicMock()
        blob.name = name
        blob.download_as_text.return_value = text
        return blob

    def test_reads_jsonl_and_preserves_echoed_key(self, provider, mock_storage):
        lines = "\n".join(
            json.dumps({"key": k, "response": {"text": t}})
            for k, t in (("row_3", "a"), ("row_7", "b"))
        )
        mock_storage.list_blobs.return_value = [
            self._blob("out/predictions.jsonl", lines),
            self._blob("out/_SUCCESS", "ignore-me"),  # non-jsonl skipped
        ]

        results = provider.download_batch_results("gs://b/out/")
        assert [r["custom_id"] for r in results] == ["row_3", "row_7"]
        assert results[0]["response"] == {"text": "a"}
        assert results[0]["error"] is None

    def test_falls_back_to_line_order_without_key(self, provider, mock_storage):
        lines = "\n".join(json.dumps({"response": {"text": t}}) for t in ("a", "b"))
        mock_storage.list_blobs.return_value = [
            self._blob("out/predictions.jsonl", lines)
        ]
        results = provider.download_batch_results("gs://b/out/")
        assert [r["custom_id"] for r in results] == ["0", "1"]

    def test_batch_name_resolves_dest_then_reads(self, provider, mock_genai, mock_storage):
        job = MagicMock()
        job.state.name = "JOB_STATE_SUCCEEDED"
        job.dest.gcs_uri = "gs://b/out/"
        mock_genai.batches.get.return_value = job
        mock_storage.list_blobs.return_value = [
            self._blob("out/p.jsonl", json.dumps({"key": "row_5", "error": "quota"}))
        ]

        results = provider.download_batch_results("batches/123")
        assert results[0]["custom_id"] == "row_5"
        assert results[0]["error"] == "quota"
        assert results[0]["response"] is None

    def test_incomplete_job_raises(self, provider, mock_genai):
        job = MagicMock()
        job.state.name = "JOB_STATE_RUNNING"
        mock_genai.batches.get.return_value = job
        with pytest.raises(ValueError, match="not complete"):
            provider.download_batch_results("batches/123")


class TestFileIO:
    def test_upload_file_returns_gs_uri(self, provider, mock_storage):
        uri = provider.upload_file('{"x":1}')
        assert uri.startswith(f"gs://{_BUCKET}/batch-input/")
        blob = mock_storage.bucket.return_value.blob.return_value
        blob.upload_from_string.assert_called_once()
        assert blob.upload_from_string.call_args.kwargs["content_type"] == (
            "application/jsonl"
        )

    def test_download_file_reads_text(self, provider, mock_storage):
        mock_storage.bucket.return_value.blob.return_value.download_as_text.return_value = (
            "hello"
        )
        assert provider.download_file("gs://b/k.jsonl") == "hello"


class TestFromCredentials:
    _CRED = {
        "project_id": "proj",
        "location": "us-central1",
        "gcs_bucket": _BUCKET,
        "sa_key": {"type": "service_account", "project_id": "proj"},
    }

    def test_builds_provider(self):
        with (
            patch("app.core.batch.vertex.service_account") as sa,
            patch("app.core.batch.vertex.genai.Client") as genai_client,
            patch("app.core.batch.vertex.gcs.Client") as gcs_client,
        ):
            provider = VertexBatchProvider.from_credentials(self._CRED)
        assert isinstance(provider, VertexBatchProvider)
        sa.Credentials.from_service_account_info.assert_called_once()
        assert genai_client.call_args.kwargs["vertexai"] is True
        gcs_client.assert_called_once()

    @pytest.mark.parametrize(
        "drop", ["project_id", "location", "gcs_bucket", "sa_key"]
    )
    def test_missing_field_raises(self, drop):
        cred = {k: v for k, v in self._CRED.items() if k != drop}
        with pytest.raises(ValueError, match=drop):
            VertexBatchProvider.from_credentials(cred)
