import json
from unittest.mock import MagicMock

import pytest

from app.core.batch.openai import OpenAIBatchProvider
from app.tests.utils.openai import create_mock_batch


class TestOpenAIBatchProvider:
    """Test cases for OpenAIBatchProvider."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_openai_client):
        """Create an OpenAIBatchProvider instance with mock client."""
        return OpenAIBatchProvider(client=mock_openai_client)

    def test_initialization(self, mock_openai_client):
        """Test that provider initializes correctly."""
        provider = OpenAIBatchProvider(client=mock_openai_client)
        assert provider.client == mock_openai_client

    def test_create_batch_success(self, provider, mock_openai_client):
        """Test successful batch creation."""
        jsonl_data = [
            {"custom_id": "req-1", "method": "POST", "url": "/v1/responses"},
            {"custom_id": "req-2", "method": "POST", "url": "/v1/responses"},
        ]
        config = {
            "endpoint": "/v1/responses",
            "description": "Test batch",
            "completion_window": "24h",
        }

        mock_file_response = MagicMock()
        mock_file_response.id = "file-abc123"
        mock_openai_client.files.create.return_value = mock_file_response

        mock_batch = MagicMock()
        mock_batch.id = "batch-xyz789"
        mock_batch.status = "validating"
        mock_openai_client.batches.create.return_value = mock_batch

        result = provider.create_batch(jsonl_data, config)

        mock_openai_client.files.create.assert_called_once()
        file_call_args = mock_openai_client.files.create.call_args
        assert file_call_args[1]["purpose"] == "batch"

        mock_openai_client.batches.create.assert_called_once_with(
            input_file_id="file-abc123",
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": "Test batch"},
        )

        assert result["provider_batch_id"] == "batch-xyz789"
        assert result["provider_file_id"] == "file-abc123"
        assert result["provider_status"] == "validating"
        assert result["total_items"] == 2

    def test_create_batch_with_default_config(self, provider, mock_openai_client):
        """Test batch creation with default configuration values."""
        jsonl_data = [{"custom_id": "req-1"}]
        config = {}

        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_openai_client.files.create.return_value = mock_file_response

        mock_batch = MagicMock()
        mock_batch.id = "batch-456"
        mock_batch.status = "validating"
        mock_openai_client.batches.create.return_value = mock_batch

        result = provider.create_batch(jsonl_data, config)

        mock_openai_client.batches.create.assert_called_once_with(
            input_file_id="file-123",
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": "LLM batch job"},
        )

        assert result["total_items"] == 1

    def test_create_batch_file_upload_error(self, provider, mock_openai_client):
        """Test handling of file upload error during batch creation."""
        jsonl_data = [{"custom_id": "req-1"}]
        config = {"endpoint": "/v1/responses"}

        mock_openai_client.files.create.side_effect = Exception("File upload failed")

        with pytest.raises(Exception) as exc_info:
            provider.create_batch(jsonl_data, config)

        assert "File upload failed" in str(exc_info.value)

    def test_create_batch_batch_creation_error(self, provider, mock_openai_client):
        """Test handling of batch creation error."""
        jsonl_data = [{"custom_id": "req-1"}]
        config = {"endpoint": "/v1/responses"}

        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_openai_client.files.create.return_value = mock_file_response

        mock_openai_client.batches.create.side_effect = Exception(
            "Batch creation failed"
        )

        with pytest.raises(Exception) as exc_info:
            provider.create_batch(jsonl_data, config)

        assert "Batch creation failed" in str(exc_info.value)

    def test_get_batch_status_completed(self, provider, mock_openai_client):
        """Test getting status of a completed batch."""
        batch_id = "batch-xyz789"

        mock_batch = create_mock_batch(
            batch_id=batch_id,
            status="completed",
            output_file_id="output-file-123",
            error_file_id=None,
            total=100,
            completed=100,
            failed=0,
        )
        mock_openai_client.batches.retrieve.return_value = mock_batch

        result = provider.get_batch_status(batch_id)

        mock_openai_client.batches.retrieve.assert_called_once_with(batch_id)
        assert result["provider_status"] == "completed"
        assert result["provider_output_file_id"] == "output-file-123"
        assert result["error_file_id"] is None
        assert result["request_counts"]["total"] == 100
        assert result["request_counts"]["completed"] == 100
        assert result["request_counts"]["failed"] == 0
        assert "error_message" not in result

    def test_get_batch_status_in_progress(self, provider, mock_openai_client):
        """Test getting status of an in-progress batch."""
        batch_id = "batch-xyz789"

        mock_batch = create_mock_batch(
            batch_id=batch_id,
            status="in_progress",
            output_file_id=None,
            total=100,
            completed=45,
            failed=0,
        )
        mock_openai_client.batches.retrieve.return_value = mock_batch

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "in_progress"
        assert result["provider_output_file_id"] is None
        assert result["request_counts"]["completed"] == 45
        assert "error_message" not in result

    def test_get_batch_status_failed(self, provider, mock_openai_client):
        """Test getting status of a failed batch."""
        batch_id = "batch-xyz789"

        mock_batch = create_mock_batch(
            batch_id=batch_id,
            status="failed",
            output_file_id=None,
            error_file_id="error-file-456",
            total=100,
            completed=50,
            failed=50,
        )
        mock_openai_client.batches.retrieve.return_value = mock_batch

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "failed"
        assert result["error_file_id"] == "error-file-456"
        assert "error_message" in result
        assert "Batch failed" in result["error_message"]
        assert "error-file-456" in result["error_message"]

    def test_get_batch_status_expired(self, provider, mock_openai_client):
        """Test getting status of an expired batch."""
        batch_id = "batch-xyz789"

        mock_batch = create_mock_batch(
            batch_id=batch_id,
            status="expired",
            output_file_id=None,
            total=100,
            completed=0,
            failed=0,
        )
        mock_openai_client.batches.retrieve.return_value = mock_batch

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "expired"
        assert "error_message" in result
        assert "Batch expired" in result["error_message"]

    def test_get_batch_status_cancelled(self, provider, mock_openai_client):
        """Test getting status of a cancelled batch."""
        batch_id = "batch-xyz789"

        mock_batch = create_mock_batch(
            batch_id=batch_id,
            status="cancelled",
            output_file_id=None,
            error_file_id="error-file-789",
            total=100,
            completed=25,
            failed=0,
        )
        mock_openai_client.batches.retrieve.return_value = mock_batch

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "cancelled"
        assert "error_message" in result
        assert "Batch cancelled" in result["error_message"]
        assert "error-file-789" in result["error_message"]

    def test_get_batch_status_error(self, provider, mock_openai_client):
        """Test handling of error when retrieving batch status."""
        batch_id = "batch-xyz789"

        mock_openai_client.batches.retrieve.side_effect = Exception(
            "API connection failed"
        )

        with pytest.raises(Exception) as exc_info:
            provider.get_batch_status(batch_id)

        assert "API connection failed" in str(exc_info.value)

    def test_download_batch_results_success(self, provider, mock_openai_client):
        """Test successful download of batch results."""
        output_file_id = "output-file-123"

        jsonl_content = (
            '{"custom_id":"req-1","response":{"status_code":200,"body":{"result":"success"}}}\n'
            '{"custom_id":"req-2","response":{"status_code":200,"body":{"result":"success"}}}'
        )

        mock_file_content = MagicMock()
        mock_file_content.read.return_value = jsonl_content.encode("utf-8")
        mock_openai_client.files.content.return_value = mock_file_content

        results = provider.download_batch_results(output_file_id)

        mock_openai_client.files.content.assert_called_once_with(output_file_id)
        assert len(results) == 2
        assert results[0]["custom_id"] == "req-1"
        assert results[0]["response"]["status_code"] == 200
        assert results[1]["custom_id"] == "req-2"

    def test_download_batch_results_with_errors(self, provider, mock_openai_client):
        """Test downloading batch results that contain errors."""
        output_file_id = "output-file-123"

        jsonl_content = (
            '{"custom_id":"req-1","response":{"status_code":200,"body":{"result":"success"}}}\n'
            '{"custom_id":"req-2","error":{"message":"Invalid request","code":"400"}}'
        )

        mock_file_content = MagicMock()
        mock_file_content.read.return_value = jsonl_content.encode("utf-8")
        mock_openai_client.files.content.return_value = mock_file_content

        results = provider.download_batch_results(output_file_id)

        assert len(results) == 2
        assert results[0]["custom_id"] == "req-1"
        assert "response" in results[0]
        assert results[1]["custom_id"] == "req-2"
        assert "error" in results[1]
        assert results[1]["error"]["code"] == "400"

    def test_download_batch_results_malformed_json(self, provider, mock_openai_client):
        """Test handling of malformed JSON in batch results."""
        output_file_id = "output-file-123"

        jsonl_content = (
            '{"custom_id":"req-1","response":{"status_code":200}}\n'
            "this is not valid json\n"
            '{"custom_id":"req-3","response":{"status_code":200}}'
        )

        mock_file_content = MagicMock()
        mock_file_content.read.return_value = jsonl_content.encode("utf-8")
        mock_openai_client.files.content.return_value = mock_file_content

        results = provider.download_batch_results(output_file_id)

        assert len(results) == 2
        assert results[0]["custom_id"] == "req-1"
        assert results[1]["custom_id"] == "req-3"

    def test_download_batch_results_empty_file(self, provider, mock_openai_client):
        """Test downloading an empty results file."""
        output_file_id = "output-file-123"

        mock_file_content = MagicMock()
        mock_file_content.read.return_value = b""
        mock_openai_client.files.content.return_value = mock_file_content

        results = provider.download_batch_results(output_file_id)

        assert len(results) == 0

    def test_download_batch_results_error(self, provider, mock_openai_client):
        """Test handling of error when downloading batch results."""
        output_file_id = "output-file-123"

        mock_openai_client.files.content.side_effect = Exception("Download failed")

        with pytest.raises(Exception) as exc_info:
            provider.download_batch_results(output_file_id)

        assert "Download failed" in str(exc_info.value)

    def test_upload_file_success(self, provider, mock_openai_client):
        """Test successful file upload."""
        content = '{"test":"data1"}\n{"test":"data2"}'
        purpose = "batch"

        mock_file_response = MagicMock()
        mock_file_response.id = "file-abc123"
        mock_openai_client.files.create.return_value = mock_file_response

        file_id = provider.upload_file(content, purpose)

        mock_openai_client.files.create.assert_called_once()
        call_args = mock_openai_client.files.create.call_args
        assert call_args[1]["purpose"] == purpose
        uploaded_file = call_args[1]["file"]
        assert uploaded_file[0] == "batch_input.jsonl"
        assert uploaded_file[1] == content.encode("utf-8")
        assert file_id == "file-abc123"

    def test_upload_file_with_default_purpose(self, provider, mock_openai_client):
        """Test file upload with default purpose."""
        content = '{"test":"data"}'

        mock_file_response = MagicMock()
        mock_file_response.id = "file-xyz789"
        mock_openai_client.files.create.return_value = mock_file_response

        file_id = provider.upload_file(content)

        call_args = mock_openai_client.files.create.call_args
        assert call_args[1]["purpose"] == "batch"
        assert file_id == "file-xyz789"

    def test_upload_file_error(self, provider, mock_openai_client):
        """Test handling of error during file upload."""
        content = '{"test":"data"}'

        mock_openai_client.files.create.side_effect = Exception("Upload quota exceeded")

        with pytest.raises(Exception) as exc_info:
            provider.upload_file(content)

        assert "Upload quota exceeded" in str(exc_info.value)

    def test_download_file_success(self, provider, mock_openai_client):
        """Test successful file download."""
        file_id = "file-abc123"
        expected_content = '{"custom_id":"req-1","data":"test"}'

        mock_file_content = MagicMock()
        mock_file_content.read.return_value = expected_content.encode("utf-8")
        mock_openai_client.files.content.return_value = mock_file_content

        content = provider.download_file(file_id)

        mock_openai_client.files.content.assert_called_once_with(file_id)
        assert content == expected_content

    def test_download_file_unicode_content(self, provider, mock_openai_client):
        """Test downloading file with unicode content."""
        file_id = "file-abc123"
        expected_content = '{"text":"Hello 世界 🌍"}'

        mock_file_content = MagicMock()
        mock_file_content.read.return_value = expected_content.encode("utf-8")
        mock_openai_client.files.content.return_value = mock_file_content

        content = provider.download_file(file_id)

        assert content == expected_content
        assert "世界" in content
        assert "🌍" in content

    def test_download_file_error(self, provider, mock_openai_client):
        """Test handling of error during file download."""
        file_id = "file-abc123"

        mock_openai_client.files.content.side_effect = Exception("File not found")

        with pytest.raises(Exception) as exc_info:
            provider.download_file(file_id)

        assert "File not found" in str(exc_info.value)

    def test_create_batch_jsonl_formatting(self, provider, mock_openai_client):
        """Test that JSONL data is properly formatted during batch creation."""
        jsonl_data = [
            {"custom_id": "req-1", "method": "POST"},
            {"custom_id": "req-2", "method": "GET"},
        ]
        config = {"endpoint": "/v1/responses"}

        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_openai_client.files.create.return_value = mock_file_response

        mock_batch = MagicMock()
        mock_batch.id = "batch-456"
        mock_batch.status = "validating"
        mock_openai_client.batches.create.return_value = mock_batch

        provider.create_batch(jsonl_data, config)

        call_args = mock_openai_client.files.create.call_args
        uploaded_content = call_args[1]["file"][1].decode("utf-8")
        lines = uploaded_content.split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["custom_id"] == "req-1"
        assert json.loads(lines[1])["custom_id"] == "req-2"
