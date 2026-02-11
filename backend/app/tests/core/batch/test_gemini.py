"""Test cases for GeminiBatchProvider."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.batch.gemini import (
    BatchJobState,
    GeminiBatchProvider,
    create_stt_batch_requests,
)


class TestGeminiBatchProvider:
    """Test cases for GeminiBatchProvider."""

    @pytest.fixture
    def mock_genai_client(self):
        """Create a mock Gemini client."""
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_genai_client):
        """Create a GeminiBatchProvider instance with mock client."""
        return GeminiBatchProvider(client=mock_genai_client)

    @pytest.fixture
    def provider_with_model(self, mock_genai_client):
        """Create a GeminiBatchProvider with custom model."""
        return GeminiBatchProvider(
            client=mock_genai_client, model="models/gemini-2.5-flash"
        )

    def test_initialization_default_model(self, mock_genai_client):
        """Test that provider initializes with default model."""
        provider = GeminiBatchProvider(client=mock_genai_client)
        assert provider._client == mock_genai_client
        assert provider._model == "models/gemini-2.5-pro"

    def test_initialization_custom_model(self, mock_genai_client):
        """Test that provider initializes with custom model."""
        provider = GeminiBatchProvider(
            client=mock_genai_client, model="models/gemini-2.5-flash"
        )
        assert provider._model == "models/gemini-2.5-flash"

    def test_create_batch_success(self, provider, mock_genai_client):
        """Test successful batch creation."""
        jsonl_data = [
            {"key": "req-1", "request": {"contents": [{"parts": [{"text": "test"}]}]}},
            {"key": "req-2", "request": {"contents": [{"parts": [{"text": "test2"}]}]}},
        ]
        config = {
            "display_name": "test-batch",
            "model": "models/gemini-2.5-pro",
        }

        # Mock file upload
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/uploaded-123"
        mock_genai_client.files.upload.return_value = mock_uploaded_file

        # Mock batch creation
        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/batch-xyz789"
        mock_batch_job.state.name = "JOB_STATE_PENDING"
        mock_genai_client.batches.create.return_value = mock_batch_job

        with patch("tempfile.NamedTemporaryFile"):
            with patch("os.unlink"):
                result = provider.create_batch(jsonl_data, config)

        assert result["provider_batch_id"] == "batches/batch-xyz789"
        assert result["provider_file_id"] == "files/uploaded-123"
        assert result["provider_status"] == "JOB_STATE_PENDING"
        assert result["total_items"] == 2

    def test_create_batch_with_default_config(self, provider, mock_genai_client):
        """Test batch creation with default configuration values."""
        jsonl_data = [{"key": "req-1", "request": {}}]
        config = {}

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/uploaded-456"
        mock_genai_client.files.upload.return_value = mock_uploaded_file

        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/batch-123"
        mock_batch_job.state.name = "JOB_STATE_PENDING"
        mock_genai_client.batches.create.return_value = mock_batch_job

        with patch("tempfile.NamedTemporaryFile"):
            with patch("os.unlink"):
                result = provider.create_batch(jsonl_data, config)

        assert result["total_items"] == 1
        mock_genai_client.batches.create.assert_called_once()

    def test_create_batch_file_upload_error(self, provider, mock_genai_client):
        """Test handling of file upload error during batch creation."""
        jsonl_data = [{"key": "req-1", "request": {}}]
        config = {"display_name": "test"}

        mock_genai_client.files.upload.side_effect = Exception("File upload failed")

        with patch("tempfile.NamedTemporaryFile"):
            with patch("os.unlink"):
                with pytest.raises(Exception) as exc_info:
                    provider.create_batch(jsonl_data, config)

        assert "File upload failed" in str(exc_info.value)

    def test_create_batch_batch_creation_error(self, provider, mock_genai_client):
        """Test handling of batch creation error."""
        jsonl_data = [{"key": "req-1", "request": {}}]
        config = {"display_name": "test"}

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/uploaded-123"
        mock_genai_client.files.upload.return_value = mock_uploaded_file
        mock_genai_client.batches.create.side_effect = Exception(
            "Batch creation failed"
        )

        with patch("tempfile.NamedTemporaryFile"):
            with patch("os.unlink"):
                with pytest.raises(Exception) as exc_info:
                    provider.create_batch(jsonl_data, config)

        assert "Batch creation failed" in str(exc_info.value)

    def test_get_batch_status_pending(self, provider, mock_genai_client):
        """Test getting status of a pending batch."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_PENDING"
        mock_genai_client.batches.get.return_value = mock_batch_job

        result = provider.get_batch_status(batch_id)

        mock_genai_client.batches.get.assert_called_once_with(name=batch_id)
        assert result["provider_status"] == "JOB_STATE_PENDING"
        assert result["provider_output_file_id"] == batch_id
        assert "error_message" not in result

    def test_get_batch_status_succeeded(self, provider, mock_genai_client):
        """Test getting status of a succeeded batch."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_genai_client.batches.get.return_value = mock_batch_job

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "JOB_STATE_SUCCEEDED"
        assert "error_message" not in result

    def test_get_batch_status_failed(self, provider, mock_genai_client):
        """Test getting status of a failed batch."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_FAILED"
        mock_genai_client.batches.get.return_value = mock_batch_job

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "JOB_STATE_FAILED"
        assert "error_message" in result
        assert "Batch JOB_STATE_FAILED" in result["error_message"]

    def test_get_batch_status_cancelled(self, provider, mock_genai_client):
        """Test getting status of a cancelled batch."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_CANCELLED"
        mock_genai_client.batches.get.return_value = mock_batch_job

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "JOB_STATE_CANCELLED"
        assert "error_message" in result

    def test_get_batch_status_expired(self, provider, mock_genai_client):
        """Test getting status of an expired batch."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_EXPIRED"
        mock_genai_client.batches.get.return_value = mock_batch_job

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "JOB_STATE_EXPIRED"
        assert "error_message" in result

    def test_get_batch_status_error(self, provider, mock_genai_client):
        """Test handling of error when retrieving batch status."""
        batch_id = "batches/batch-xyz789"

        mock_genai_client.batches.get.side_effect = Exception("API connection failed")

        with pytest.raises(Exception) as exc_info:
            provider.get_batch_status(batch_id)

        assert "API connection failed" in str(exc_info.value)

    def test_download_batch_results_success(self, provider, mock_genai_client):
        """Test successful download of batch results."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_batch_job.dest = MagicMock()
        mock_batch_job.dest.file_name = "files/output-123"
        mock_genai_client.batches.get.return_value = mock_batch_job

        jsonl_content = (
            '{"key":"req-1","response":{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}}\n'
            '{"key":"req-2","response":{"candidates":[{"content":{"parts":[{"text":"World"}]}}]}}'
        )
        mock_genai_client.files.download.return_value = jsonl_content.encode("utf-8")

        results = provider.download_batch_results(batch_id)

        mock_genai_client.batches.get.assert_called_once_with(name=batch_id)
        mock_genai_client.files.download.assert_called_once_with(
            file="files/output-123"
        )
        assert len(results) == 2
        assert results[0]["custom_id"] == "req-1"
        assert results[0]["response"]["text"] == "Hello"
        assert results[1]["custom_id"] == "req-2"
        assert results[1]["response"]["text"] == "World"

    def test_download_batch_results_with_direct_text_response(
        self, provider, mock_genai_client
    ):
        """Test downloading results with direct text in response."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_batch_job.dest = MagicMock()
        mock_batch_job.dest.file_name = "files/output-123"
        mock_genai_client.batches.get.return_value = mock_batch_job

        jsonl_content = '{"key":"req-1","response":{"text":"Direct text"}}'
        mock_genai_client.files.download.return_value = jsonl_content.encode("utf-8")

        results = provider.download_batch_results(batch_id)

        assert len(results) == 1
        assert results[0]["response"]["text"] == "Direct text"

    def test_download_batch_results_with_errors(self, provider, mock_genai_client):
        """Test downloading batch results that contain errors."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_batch_job.dest = MagicMock()
        mock_batch_job.dest.file_name = "files/output-123"
        mock_genai_client.batches.get.return_value = mock_batch_job

        jsonl_content = (
            '{"key":"req-1","response":{"text":"Success"}}\n'
            '{"key":"req-2","error":{"message":"Invalid request"}}'
        )
        mock_genai_client.files.download.return_value = jsonl_content.encode("utf-8")

        results = provider.download_batch_results(batch_id)

        assert len(results) == 2
        assert results[0]["custom_id"] == "req-1"
        assert results[0]["response"] is not None
        assert results[0]["error"] is None
        assert results[1]["custom_id"] == "req-2"
        assert results[1]["response"] is None
        assert "Invalid request" in results[1]["error"]

    def test_download_batch_results_batch_not_complete(
        self, provider, mock_genai_client
    ):
        """Test error when trying to download results from incomplete batch."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_RUNNING"
        mock_genai_client.batches.get.return_value = mock_batch_job

        with pytest.raises(ValueError) as exc_info:
            provider.download_batch_results(batch_id)

        assert "Batch job not complete" in str(exc_info.value)

    def test_download_batch_results_malformed_json(self, provider, mock_genai_client):
        """Test handling of malformed JSON in batch results."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_batch_job.dest = MagicMock()
        mock_batch_job.dest.file_name = "files/output-123"
        mock_genai_client.batches.get.return_value = mock_batch_job

        jsonl_content = (
            '{"key":"req-1","response":{"text":"Valid"}}\n'
            "this is not valid json\n"
            '{"key":"req-3","response":{"text":"Also valid"}}'
        )
        mock_genai_client.files.download.return_value = jsonl_content.encode("utf-8")

        results = provider.download_batch_results(batch_id)

        # Should skip the malformed line and process the rest
        assert len(results) == 2
        assert results[0]["custom_id"] == "req-1"
        assert results[1]["custom_id"] == "req-3"

    def test_download_batch_results_no_dest_file(self, provider, mock_genai_client):
        """Test handling when batch has no destination file."""
        batch_id = "batches/batch-xyz789"

        mock_batch_job = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_batch_job.dest = None
        mock_genai_client.batches.get.return_value = mock_batch_job

        results = provider.download_batch_results(batch_id)

        assert len(results) == 0

    def test_upload_file_success(self, provider, mock_genai_client):
        """Test successful file upload."""
        content = '{"key":"req-1","request":{}}\n{"key":"req-2","request":{}}'

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/uploaded-abc123"
        mock_genai_client.files.upload.return_value = mock_uploaded_file

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test.jsonl"
            mock_temp.return_value.__enter__.return_value = mock_temp_file

            with patch("os.unlink"):
                file_name = provider.upload_file(content, purpose="batch")

        assert file_name == "files/uploaded-abc123"
        mock_genai_client.files.upload.assert_called_once()

    def test_upload_file_error(self, provider, mock_genai_client):
        """Test handling of error during file upload."""
        content = '{"key":"req-1"}'

        mock_genai_client.files.upload.side_effect = Exception("Upload quota exceeded")

        with patch("tempfile.NamedTemporaryFile"):
            with patch("os.unlink"):
                with pytest.raises(Exception) as exc_info:
                    provider.upload_file(content)

        assert "Upload quota exceeded" in str(exc_info.value)

    def test_download_file_success(self, provider, mock_genai_client):
        """Test successful file download."""
        file_id = "files/abc123"
        expected_content = '{"key":"req-1","response":{"text":"test"}}'

        mock_genai_client.files.download.return_value = expected_content.encode("utf-8")

        content = provider.download_file(file_id)

        mock_genai_client.files.download.assert_called_once_with(file=file_id)
        assert content == expected_content

    def test_download_file_unicode_content(self, provider, mock_genai_client):
        """Test downloading file with unicode content."""
        file_id = "files/abc123"
        expected_content = '{"text":"Hello 世界 🌍"}'

        mock_genai_client.files.download.return_value = expected_content.encode("utf-8")

        content = provider.download_file(file_id)

        assert content == expected_content
        assert "世界" in content
        assert "🌍" in content

    def test_download_file_error(self, provider, mock_genai_client):
        """Test handling of error during file download."""
        file_id = "files/abc123"

        mock_genai_client.files.download.side_effect = Exception("File not found")

        with pytest.raises(Exception) as exc_info:
            provider.download_file(file_id)

        assert "File not found" in str(exc_info.value)


class TestBatchJobState:
    """Test cases for BatchJobState enum."""

    def test_batch_job_states(self):
        """Test that all batch job states have correct values."""
        assert BatchJobState.PENDING.value == "JOB_STATE_PENDING"
        assert BatchJobState.RUNNING.value == "JOB_STATE_RUNNING"
        assert BatchJobState.SUCCEEDED.value == "JOB_STATE_SUCCEEDED"
        assert BatchJobState.FAILED.value == "JOB_STATE_FAILED"
        assert BatchJobState.CANCELLED.value == "JOB_STATE_CANCELLED"
        assert BatchJobState.EXPIRED.value == "JOB_STATE_EXPIRED"


class TestCreateSTTBatchRequests:
    """Test cases for create_stt_batch_requests function."""

    def test_create_requests_with_keys(self):
        """Test creating batch requests with custom keys."""
        signed_urls = [
            "https://bucket.s3.amazonaws.com/audio1.mp3?signature=abc",
            "https://bucket.s3.amazonaws.com/audio2.wav?signature=def",
        ]
        prompt = "Transcribe this audio file."
        keys = ["sample-1", "sample-2"]

        requests = create_stt_batch_requests(signed_urls, prompt, keys=keys)

        assert len(requests) == 2
        assert requests[0]["key"] == "sample-1"
        assert requests[1]["key"] == "sample-2"

        # Verify structure
        assert "request" in requests[0]
        assert "contents" in requests[0]["request"]
        contents = requests[0]["request"]["contents"]
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert len(contents[0]["parts"]) == 2
        assert contents[0]["parts"][0]["text"] == prompt
        assert "file_data" in contents[0]["parts"][1]

    def test_create_requests_without_keys(self):
        """Test creating batch requests without keys (auto-generated)."""
        signed_urls = [
            "https://bucket.s3.amazonaws.com/audio.mp3?signature=xyz",
        ]
        prompt = "Transcribe."

        requests = create_stt_batch_requests(signed_urls, prompt)

        assert len(requests) == 1
        assert requests[0]["key"] == "0"

    def test_create_requests_mime_type_detection(self):
        """Test that MIME types are correctly detected from URLs."""
        signed_urls = [
            "https://bucket.s3.amazonaws.com/audio.mp3?sig=1",
            "https://bucket.s3.amazonaws.com/audio.wav?sig=2",
            "https://bucket.s3.amazonaws.com/audio.m4a?sig=3",
        ]
        prompt = "Transcribe."

        requests = create_stt_batch_requests(signed_urls, prompt)

        assert (
            requests[0]["request"]["contents"][0]["parts"][1]["file_data"]["mime_type"]
            == "audio/mpeg"
        )
        assert (
            requests[1]["request"]["contents"][0]["parts"][1]["file_data"]["mime_type"]
            == "audio/x-wav"
        )
        # .m4a can return different MIME types depending on the system
        m4a_mime = requests[2]["request"]["contents"][0]["parts"][1]["file_data"][
            "mime_type"
        ]
        assert m4a_mime in ("audio/mp4", "audio/mp4a-latm", "audio/x-m4a")

    def test_create_requests_key_length_mismatch(self):
        """Test that mismatched keys and URLs raise error."""
        signed_urls = [
            "https://example.com/audio1.mp3",
            "https://example.com/audio2.mp3",
        ]
        keys = ["only-one-key"]
        prompt = "Transcribe."

        with pytest.raises(ValueError) as exc_info:
            create_stt_batch_requests(signed_urls, prompt, keys=keys)

        assert "Length of keys" in str(exc_info.value)

    def test_create_requests_file_uri_preserved(self):
        """Test that signed URLs are preserved in file_uri."""
        signed_url = "https://bucket.s3.amazonaws.com/audio.mp3?X-Amz-Signature=abc123&X-Amz-Expires=3600"
        prompt = "Transcribe."

        requests = create_stt_batch_requests([signed_url], prompt)

        file_uri = requests[0]["request"]["contents"][0]["parts"][1]["file_data"][
            "file_uri"
        ]
        assert file_uri == signed_url
        assert "X-Amz-Signature" in file_uri


class TestExtractTextFromResponseDict:
    """Test cases for _extract_text_from_response_dict static method."""

    def test_extract_direct_text(self):
        """Test extracting text from direct text field."""
        response = {"text": "Hello world"}
        text = GeminiBatchProvider._extract_text_from_response_dict(response)
        assert text == "Hello world"

    def test_extract_from_candidates(self):
        """Test extracting text from candidates structure."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Part 1"},
                            {"text": " Part 2"},
                        ]
                    }
                }
            ]
        }
        text = GeminiBatchProvider._extract_text_from_response_dict(response)
        assert text == "Part 1 Part 2"

    def test_extract_empty_response(self):
        """Test extracting text from empty response."""
        response = {}
        text = GeminiBatchProvider._extract_text_from_response_dict(response)
        assert text == ""

    def test_extract_multiple_candidates(self):
        """Test extracting text from multiple candidates."""
        response = {
            "candidates": [
                {"content": {"parts": [{"text": "First"}]}},
                {"content": {"parts": [{"text": "Second"}]}},
            ]
        }
        text = GeminiBatchProvider._extract_text_from_response_dict(response)
        assert text == "FirstSecond"


class TestExtractTextFromResponse:
    """Test cases for _extract_text_from_response static method (object version)."""

    def test_extract_direct_text_attribute(self):
        """Test extracting text when response has .text attribute."""
        response = MagicMock()
        response.text = "Hello from text attribute"
        text = GeminiBatchProvider._extract_text_from_response(response)
        assert text == "Hello from text attribute"

    def test_extract_from_candidates_structure(self):
        """Test extracting text from candidates when no .text attribute."""
        # Create mock without .text attribute
        response = MagicMock(spec=[])
        del response.text  # Ensure no text attribute

        # Create candidates structure
        part1 = MagicMock()
        part1.text = "Part 1"
        part2 = MagicMock()
        part2.text = " Part 2"

        content = MagicMock()
        content.parts = [part1, part2]

        candidate = MagicMock()
        candidate.content = content

        response.candidates = [candidate]

        text = GeminiBatchProvider._extract_text_from_response(response)
        assert text == "Part 1 Part 2"

    def test_extract_empty_response_no_text_no_candidates(self):
        """Test extracting text from response with no text and no candidates."""
        response = MagicMock(spec=[])
        del response.text
        del response.candidates

        text = GeminiBatchProvider._extract_text_from_response(response)
        assert text == ""


class TestCreateSttBatchRequestsMimeTypeFallback:
    """Test cases for create_stt_batch_requests MIME type fallback."""

    def test_unknown_mime_type_defaults_to_audio_mpeg(self):
        """Test that unknown file extensions default to audio/mpeg."""
        # URL with no recognizable audio extension
        signed_urls = ["https://bucket.s3.amazonaws.com/audio.unknown?signature=xyz"]
        prompt = "Transcribe this audio."

        with patch("app.core.batch.gemini.get_mime_from_url", return_value=None):
            requests = create_stt_batch_requests(signed_urls, prompt)

        assert len(requests) == 1
        # Check that the request was created with default mime type
        # parts[0] is text prompt, parts[1] is file_data
        file_data = requests[0]["request"]["contents"][0]["parts"][1]["file_data"]
        assert file_data["mime_type"] == "audio/mpeg"
