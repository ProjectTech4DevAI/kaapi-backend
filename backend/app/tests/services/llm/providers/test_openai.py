"""
Tests for the OpenAI provider.
"""

import pytest
from unittest.mock import MagicMock

import openai

from app.models.llm import (
    NativeCompletionConfig,
    QueryParams,
)
from app.models.llm.request import ConversationConfig

from app.services.llm.providers.oai import OpenAIProvider
from app.tests.utils.openai import mock_openai_response


class TestOpenAIProvider:
    """Test cases for the OpenAIProvider class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_client):
        """Create an OpenAIProvider instance with mock client."""
        return OpenAIProvider(client=mock_client)

    @pytest.fixture
    def completion_config(self):
        """Create a basic completion config."""
        return NativeCompletionConfig(
            provider="openai-native",
            type="text",
            params={"model": "gpt-4"},
        )

    @pytest.fixture
    def query_params(self):
        """Create basic query parameters."""
        return QueryParams(input="Test query")

    def test_provider_initialization(self, mock_client):
        """Test OpenAIProvider initializes correctly."""
        provider = OpenAIProvider(client=mock_client)
        assert provider.client == mock_client

    def test_execute_success_without_conversation(
        self, provider, mock_client, completion_config, query_params
    ):
        """Test successful execution without conversation handling."""
        mock_response = mock_openai_response(text="Test response", model="gpt-4")
        mock_client.responses.create.return_value = mock_response

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert error is None
        assert result is not None
        assert result.response.output.content.value == mock_response.output_text
        assert result.response.model == mock_response.model
        assert result.response.provider == "openai-native"
        assert result.response.conversation_id is None
        assert result.usage.input_tokens == mock_response.usage.input_tokens
        assert result.usage.output_tokens == mock_response.usage.output_tokens
        assert result.usage.total_tokens == mock_response.usage.total_tokens
        assert result.provider_raw_response is None

        mock_client.responses.create.assert_called_once()

    def test_execute_with_existing_conversation_id(
        self, provider, mock_client, completion_config, query_params
    ):
        """Test execution with existing conversation ID."""
        conversation_id = "conv_123"
        query_params.conversation = ConversationConfig(id=conversation_id)

        mock_response = mock_openai_response(
            text="Response with conversation",
            model="gpt-4",
            conversation_id=conversation_id,
        )
        mock_client.responses.create.return_value = mock_response

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert error is None
        assert result is not None
        assert result.response.conversation_id == conversation_id

        # Verify conversation ID was passed
        call_args = mock_client.responses.create.call_args
        assert call_args[1]["conversation"] == {"id": conversation_id}

    def test_execute_with_auto_create_conversation(
        self,
        provider,
        mock_client,
        completion_config,
        query_params,
    ):
        """Test execution with auto-create conversation."""
        new_conversation_id = "conv_auto_456"
        query_params.conversation = ConversationConfig(auto_create=True)

        mock_conversation = MagicMock()
        mock_conversation.id = new_conversation_id
        mock_client.conversations.create.return_value = mock_conversation

        mock_response = mock_openai_response(
            text="Response with auto-created conversation",
            model="gpt-4",
            conversation_id=new_conversation_id,
        )
        mock_client.responses.create.return_value = mock_response

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert error is None
        assert result is not None
        assert result.response.conversation_id == new_conversation_id

        mock_client.conversations.create.assert_called_once()

        call_args = mock_client.responses.create.call_args
        assert call_args[1]["conversation"] == {"id": new_conversation_id}

    def test_execute_with_include_provider_raw_response(
        self, provider, mock_client, completion_config, query_params
    ):
        """Test execution with include_provider_raw_response=True."""
        mock_response = mock_openai_response(
            text="Test response",
            model="gpt-4",
            conversation_id="conv_789",
        )
        mock_client.responses.create.return_value = mock_response

        result, error = provider.execute(
            completion_config,
            query_params,
            "Test query",
            include_provider_raw_response=True,
        )

        assert error is None
        assert result is not None
        assert result.provider_raw_response is not None
        assert isinstance(result.provider_raw_response, dict)
        assert result.provider_raw_response == mock_response.model_dump()

    def test_execute_with_type_error(
        self, provider, mock_client, completion_config, query_params
    ):
        """Test handling of TypeError (invalid parameters)."""
        mock_client.responses.create.side_effect = TypeError(
            "unexpected keyword argument 'invalid_param'"
        )

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert result is None
        assert error is not None
        assert "Invalid or unexpected parameter in Config" in error
        assert "unexpected keyword argument" in error

    def test_execute_with_openai_api_error(
        self, provider, mock_client, completion_config, query_params
    ):
        """Generic `openai.APIError` (no specific subclass) falls through to the
        `except openai.OpenAIError` block, which prefixes with "OpenAI error:"
        and uses the raw exception string. No wrapper helper is invoked."""
        mock_client.responses.create.side_effect = openai.APIError(
            message="API request failed",
            request=MagicMock(),
            body=None,
        )

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert result is None
        assert error is not None
        assert error.startswith("OpenAI error:")
        assert "API request failed" in error

    def test_execute_with_generic_exception(
        self, provider, mock_client, completion_config, query_params
    ):
        """Non-OpenAI exceptions land in the `except Exception` catch-all,
        prefixed with "Unexpected error:" and carrying the exception text."""
        mock_client.responses.create.side_effect = Exception("Timeout occurred")

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert result is None
        assert error is not None
        assert error.startswith("Unexpected error:")
        assert "Timeout occurred" in error

    @pytest.mark.parametrize(
        "exception_factory, expected_prefix, expected_status, original_message",
        [
            (
                lambda: openai.RateLimitError(
                    message="quota exceeded",
                    response=MagicMock(
                        status_code=429, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "OpenAI rate limit exceeded",
                429,
                "quota exceeded",
            ),
            (
                lambda: openai.AuthenticationError(
                    message="bad api key",
                    response=MagicMock(
                        status_code=401, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "OpenAI authentication failed",
                401,
                "bad api key",
            ),
            (
                lambda: openai.NotFoundError(
                    message="model not found",
                    response=MagicMock(
                        status_code=404, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "OpenAI resource not found",
                404,
                "model not found",
            ),
            (
                lambda: openai.BadRequestError(
                    message="invalid model param",
                    response=MagicMock(
                        status_code=400, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "OpenAI bad request",
                400,
                "invalid model param",
            ),
            (
                lambda: openai.UnprocessableEntityError(
                    message="cannot process",
                    response=MagicMock(
                        status_code=422, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "OpenAI unprocessable entity",
                422,
                "cannot process",
            ),
            (
                lambda: openai.InternalServerError(
                    message="upstream boom",
                    response=MagicMock(
                        status_code=500, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "OpenAI server error",
                500,
                "upstream boom",
            ),
        ],
    )
    def test_execute_specific_openai_exceptions_use_category_prefix(
        self,
        provider,
        mock_client,
        completion_config,
        query_params,
        exception_factory,
        expected_prefix,
        expected_status,
        original_message,
    ):
        """Each specific OpenAI exception with a dedicated handler emits a
        category-prefixed message including the upstream status code and the
        original error text. Structural assertions only — the remediation
        suffix can evolve without breaking the suite.
        """
        mock_client.responses.create.side_effect = exception_factory()

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert result is None
        assert error is not None
        assert error.startswith(expected_prefix), error
        assert f"code: {expected_status}" in error, error
        assert original_message in error, error

    def test_execute_with_api_timeout_error(
        self, provider, mock_client, completion_config, query_params
    ):
        """APITimeoutError doesn't expose .message — handler interpolates str(e)."""
        mock_client.responses.create.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert result is None
        assert error is not None
        assert error.startswith("OpenAI request timed out:")

    @pytest.mark.parametrize(
        "exception_factory, original_message",
        [
            (
                lambda: openai.PermissionDeniedError(
                    message="no access to model",
                    response=MagicMock(
                        status_code=403, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "no access to model",
            ),
            (
                lambda: openai.ConflictError(
                    message="resource conflict",
                    response=MagicMock(
                        status_code=409, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "resource conflict",
            ),
            (
                lambda: openai.APIConnectionError(
                    message="connection refused", request=MagicMock()
                ),
                "connection refused",
            ),
            (
                lambda: openai.APIStatusError(
                    "teapot",
                    response=MagicMock(
                        status_code=418, request=MagicMock(), headers={}
                    ),
                    body=None,
                ),
                "teapot",
            ),
        ],
    )
    def test_execute_unhandled_openai_subclasses_fall_through_to_catch_all(
        self,
        provider,
        mock_client,
        completion_config,
        query_params,
        exception_factory,
        original_message,
    ):
        """PermissionDenied, Conflict, APIConnection, and APIStatus no longer
        have dedicated except blocks. They land in the bottom-most
        `except openai.OpenAIError` handler — prefixed with the generic
        "OpenAI error" tag but still surfacing the original message.
        """
        mock_client.responses.create.side_effect = exception_factory()

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert result is None
        assert error is not None
        assert error.startswith("OpenAI error:"), error
        assert original_message in error, error

    def test_execute_with_conversation_config_without_id_or_auto_create(
        self, provider, mock_client, completion_config, query_params
    ):
        """Test that conversation is not passed if config exists but has no id or auto_create."""
        query_params.conversation = ConversationConfig()

        mock_response = mock_openai_response(text="Test response", model="gpt-4")
        mock_client.responses.create.return_value = mock_response

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert error is None
        assert result is not None

        call_args = mock_client.responses.create.call_args
        assert "conversation" not in call_args[1]

    def test_execute_merges_params_correctly(
        self, provider, mock_client, completion_config, query_params
    ):
        """Test that completion config params are merged correctly with input."""
        completion_config.params["temperature"] = 0.7
        completion_config.params["max_tokens"] = 100

        mock_response = mock_openai_response(text="Test response", model="gpt-4")
        mock_client.responses.create.return_value = mock_response

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert error is None
        assert result is not None

        # Verify params were merged
        call_args = mock_client.responses.create.call_args
        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 100
        assert call_args[1]["input"] == "Test query"

    def test_execute_with_conversation_parameter_removed_when_no_config(
        self, provider, mock_client, query_params
    ):
        """Test that conversation param is removed if it exists in config but no conversation config."""
        # Create a config with conversation in params (should be removed)
        completion_config = NativeCompletionConfig(
            provider="openai-native",
            type="text",
            params={"model": "gpt-4", "conversation": {"id": "old_conv"}},
        )

        mock_response = mock_openai_response(text="Test response", model="gpt-4")
        mock_client.responses.create.return_value = mock_response

        result, error = provider.execute(completion_config, query_params, "Test query")

        assert error is None
        assert result is not None

        # Verify old conversation was removed
        call_args = mock_client.responses.create.call_args
        assert "conversation" not in call_args[1]
