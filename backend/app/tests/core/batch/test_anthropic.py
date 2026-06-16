from unittest.mock import MagicMock

import pytest

from app.core.batch.anthropic import AnthropicBatchProvider
from app.models.llm.constants import (
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    DEFAULT_TEXT_MODELS,
)


def create_mock_batch(
    batch_id: str = "msgbatch_123",
    processing_status: str = "in_progress",
    processing: int = 0,
    succeeded: int = 0,
    errored: int = 0,
    canceled: int = 0,
    expired: int = 0,
) -> MagicMock:
    """Create a mock Anthropic MessageBatch object."""
    mock_batch = MagicMock()
    mock_batch.id = batch_id
    mock_batch.processing_status = processing_status
    mock_batch.request_counts.processing = processing
    mock_batch.request_counts.succeeded = succeeded
    mock_batch.request_counts.errored = errored
    mock_batch.request_counts.canceled = canceled
    mock_batch.request_counts.expired = expired
    return mock_batch


def create_mock_result(
    custom_id: str,
    result_type: str = "succeeded",
    message_dump: dict | None = None,
    error_dump: dict | None = None,
) -> MagicMock:
    """Create a mock Anthropic batch individual result."""
    mock_result = MagicMock()
    mock_result.custom_id = custom_id
    mock_result.result.type = result_type
    if message_dump is not None:
        mock_result.result.message.model_dump.return_value = message_dump
    if error_dump is not None:
        mock_result.result.error.model_dump.return_value = error_dump
    return mock_result


class TestAnthropicBatchProvider:
    """Test cases for AnthropicBatchProvider."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_anthropic_client):
        """Create an AnthropicBatchProvider instance with mock client."""
        return AnthropicBatchProvider(client=mock_anthropic_client)

    def test_initialization(self, mock_anthropic_client):
        """Test that provider initializes correctly."""
        provider = AnthropicBatchProvider(client=mock_anthropic_client)
        assert provider.client == mock_anthropic_client

    def test_create_batch_success(self, provider, mock_anthropic_client):
        """Test successful batch creation with inline requests."""
        jsonl_data = [
            {
                "custom_id": "req-1",
                "params": {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            },
            {
                "custom_id": "req-2",
                "params": {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi again"}],
                },
            },
        ]
        config = {}

        mock_batch = create_mock_batch(
            batch_id="msgbatch_abc123", processing_status="in_progress"
        )
        mock_anthropic_client.messages.batches.create.return_value = mock_batch

        result = provider.create_batch(jsonl_data, config)

        mock_anthropic_client.messages.batches.create.assert_called_once()
        requests = mock_anthropic_client.messages.batches.create.call_args.kwargs[
            "requests"
        ]
        assert len(requests) == 2
        assert requests[0]["custom_id"] == "req-1"
        assert requests[0]["params"]["model"] == "claude-sonnet-4-6"

        assert result["provider_batch_id"] == "msgbatch_abc123"
        assert result["provider_file_id"] is None
        assert result["provider_status"] == "in_progress"
        assert result["total_items"] == 2

    def test_create_batch_applies_config_defaults(
        self, provider, mock_anthropic_client
    ):
        """Test that model and max_tokens defaults from config are applied."""
        jsonl_data = [
            {
                "custom_id": "req-1",
                "params": {"messages": [{"role": "user", "content": "Hello"}]},
            }
        ]
        config = {"model": "claude-opus-4-8", "max_tokens": 2048}

        mock_batch = create_mock_batch()
        mock_anthropic_client.messages.batches.create.return_value = mock_batch

        provider.create_batch(jsonl_data, config)

        requests = mock_anthropic_client.messages.batches.create.call_args.kwargs[
            "requests"
        ]
        assert requests[0]["params"]["model"] == "claude-opus-4-8"
        assert requests[0]["params"]["max_tokens"] == 2048

    def test_create_batch_applies_global_defaults(
        self, provider, mock_anthropic_client
    ):
        """Test fallback to platform defaults when config has no overrides."""
        jsonl_data = [
            {
                "custom_id": "req-1",
                "params": {"messages": [{"role": "user", "content": "Hello"}]},
            }
        ]

        mock_batch = create_mock_batch()
        mock_anthropic_client.messages.batches.create.return_value = mock_batch

        provider.create_batch(jsonl_data, {})

        requests = mock_anthropic_client.messages.batches.create.call_args.kwargs[
            "requests"
        ]
        assert requests[0]["params"]["model"] == DEFAULT_TEXT_MODELS["anthropic"]
        assert requests[0]["params"]["max_tokens"] == DEFAULT_ANTHROPIC_MAX_TOKENS

    def test_create_batch_does_not_override_request_params(
        self, provider, mock_anthropic_client
    ):
        """Test that per-request model/max_tokens win over config defaults."""
        jsonl_data = [
            {
                "custom_id": "req-1",
                "params": {
                    "model": "claude-haiku-4-5",
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            }
        ]
        config = {"model": "claude-opus-4-8", "max_tokens": 2048}

        mock_batch = create_mock_batch()
        mock_anthropic_client.messages.batches.create.return_value = mock_batch

        provider.create_batch(jsonl_data, config)

        requests = mock_anthropic_client.messages.batches.create.call_args.kwargs[
            "requests"
        ]
        assert requests[0]["params"]["model"] == "claude-haiku-4-5"
        assert requests[0]["params"]["max_tokens"] == 256

    def test_create_batch_error(self, provider, mock_anthropic_client):
        """Test handling of batch creation error."""
        jsonl_data = [{"custom_id": "req-1", "params": {}}]

        mock_anthropic_client.messages.batches.create.side_effect = Exception(
            "Batch creation failed"
        )

        with pytest.raises(Exception) as exc_info:
            provider.create_batch(jsonl_data, {})

        assert "Batch creation failed" in str(exc_info.value)

    def test_get_batch_status_in_progress(self, provider, mock_anthropic_client):
        """Test getting status of an in-progress batch."""
        batch_id = "msgbatch_abc123"

        mock_batch = create_mock_batch(
            batch_id=batch_id,
            processing_status="in_progress",
            processing=55,
            succeeded=45,
        )
        mock_anthropic_client.messages.batches.retrieve.return_value = mock_batch

        result = provider.get_batch_status(batch_id)

        mock_anthropic_client.messages.batches.retrieve.assert_called_once_with(
            batch_id
        )
        assert result["provider_status"] == "in_progress"
        assert result["provider_output_file_id"] == batch_id
        assert result["request_counts"]["total"] == 100
        assert result["request_counts"]["completed"] == 45
        assert result["request_counts"]["failed"] == 0
        assert "error_message" not in result

    def test_get_batch_status_ended_success(self, provider, mock_anthropic_client):
        """Test getting status of a batch that ended with successes."""
        batch_id = "msgbatch_abc123"

        mock_batch = create_mock_batch(
            batch_id=batch_id,
            processing_status="ended",
            succeeded=98,
            errored=2,
        )
        mock_anthropic_client.messages.batches.retrieve.return_value = mock_batch

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "ended"
        assert result["request_counts"]["completed"] == 98
        assert result["request_counts"]["failed"] == 2
        assert "error_message" not in result

    def test_get_batch_status_ended_all_failed(self, provider, mock_anthropic_client):
        """Test that a batch ending with zero successes sets error_message."""
        batch_id = "msgbatch_abc123"

        mock_batch = create_mock_batch(
            batch_id=batch_id,
            processing_status="ended",
            errored=80,
            canceled=10,
            expired=10,
        )
        mock_anthropic_client.messages.batches.retrieve.return_value = mock_batch

        result = provider.get_batch_status(batch_id)

        assert result["provider_status"] == "ended"
        assert result["request_counts"]["failed"] == 100
        assert "error_message" in result
        assert "errored=80" in result["error_message"]
        assert "canceled=10" in result["error_message"]
        assert "expired=10" in result["error_message"]

    def test_get_batch_status_error(self, provider, mock_anthropic_client):
        """Test handling of error when retrieving batch status."""
        mock_anthropic_client.messages.batches.retrieve.side_effect = Exception(
            "API connection failed"
        )

        with pytest.raises(Exception) as exc_info:
            provider.get_batch_status("msgbatch_abc123")

        assert "API connection failed" in str(exc_info.value)

    def test_download_batch_results_success(self, provider, mock_anthropic_client):
        """Test successful download of batch results."""
        batch_id = "msgbatch_abc123"

        message_dump = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_results = [
            create_mock_result("req-1", "succeeded", message_dump=message_dump),
            create_mock_result("req-2", "succeeded", message_dump=message_dump),
        ]
        mock_anthropic_client.messages.batches.results.return_value = iter(mock_results)

        results = provider.download_batch_results(batch_id)

        mock_anthropic_client.messages.batches.results.assert_called_once_with(batch_id)
        assert len(results) == 2
        assert results[0]["custom_id"] == "req-1"
        assert results[0]["response"]["content"][0]["text"] == "Hello!"
        assert results[0]["error"] is None
        assert results[1]["custom_id"] == "req-2"

    def test_download_batch_results_with_errors(self, provider, mock_anthropic_client):
        """Test downloading batch results that contain errored items."""
        batch_id = "msgbatch_abc123"

        message_dump = {"content": [{"type": "text", "text": "OK"}]}
        error_dump = {"type": "invalid_request_error", "message": "Bad input"}
        mock_results = [
            create_mock_result("req-1", "succeeded", message_dump=message_dump),
            create_mock_result("req-2", "errored", error_dump=error_dump),
        ]
        mock_anthropic_client.messages.batches.results.return_value = iter(mock_results)

        results = provider.download_batch_results(batch_id)

        assert len(results) == 2
        assert results[0]["error"] is None
        assert results[1]["custom_id"] == "req-2"
        assert results[1]["response"] is None
        assert "invalid_request_error" in results[1]["error"]

    def test_download_batch_results_expired_and_canceled(
        self, provider, mock_anthropic_client
    ):
        """Test downloading batch results with expired and canceled items."""
        batch_id = "msgbatch_abc123"

        mock_results = [
            create_mock_result("req-1", "expired"),
            create_mock_result("req-2", "canceled"),
        ]
        mock_anthropic_client.messages.batches.results.return_value = iter(mock_results)

        results = provider.download_batch_results(batch_id)

        assert len(results) == 2
        assert results[0]["response"] is None
        assert results[0]["error"] == "Request expired"
        assert results[1]["error"] == "Request canceled"

    def test_download_batch_results_empty(self, provider, mock_anthropic_client):
        """Test downloading results from an empty batch."""
        mock_anthropic_client.messages.batches.results.return_value = iter([])

        results = provider.download_batch_results("msgbatch_abc123")

        assert len(results) == 0

    def test_download_batch_results_error(self, provider, mock_anthropic_client):
        """Test handling of error when downloading batch results."""
        mock_anthropic_client.messages.batches.results.side_effect = Exception(
            "Download failed"
        )

        with pytest.raises(Exception) as exc_info:
            provider.download_batch_results("msgbatch_abc123")

        assert "Download failed" in str(exc_info.value)

    def test_upload_file_not_supported(self, provider):
        """Test that upload_file raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            provider.upload_file('{"test":"data"}')

    def test_download_file_not_supported(self, provider):
        """Test that download_file raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            provider.download_file("file-123")
