# """
# Tests for the Anthropic Claude provider.
# """

# import pytest
# from unittest.mock import MagicMock
# from types import SimpleNamespace

# import anthropic

# from app.models.llm import (
#     NativeCompletionConfig,
#     QueryParams,
#     TextContent,
#     ImageContent,
#     PDFContent,
# )
# from app.services.llm.providers.base import MultiModalInput
# from app.services.llm.providers.claude import ClaudeProvider, DEFAULT_MAX_TOKENS


# def mock_claude_message(
#     text: str = "hello",
#     model: str = "claude-opus-4-7",
#     message_id: str = "msg_123",
#     input_tokens: int = 10,
#     output_tokens: int = 5,
#     extra_blocks: list | None = None,
# ) -> SimpleNamespace:
#     """Build a SimpleNamespace mimicking an anthropic Message."""
#     content = [SimpleNamespace(type="text", text=text)]
#     if extra_blocks:
#         content.extend(extra_blocks)
#     return SimpleNamespace(
#         id=message_id,
#         model=model,
#         content=content,
#         usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
#         model_dump=lambda: {"id": message_id, "model": model},
#     )


# class TestClaudeProvider:
#     @pytest.fixture
#     def mock_client(self):
#         client = MagicMock()
#         client.messages.create = MagicMock()
#         return client

#     @pytest.fixture
#     def provider(self, mock_client):
#         return ClaudeProvider(client=mock_client)

#     @pytest.fixture
#     def text_config(self):
#         return NativeCompletionConfig(
#             provider="anthropic-native",
#             type="text",
#             params={"model": "claude-opus-4-7"},
#         )

#     @pytest.fixture
#     def query_params(self):
#         return QueryParams(input="hi")

#     def test_create_client_requires_api_key(self):
#         with pytest.raises(ValueError, match="not configured"):
#             ClaudeProvider.create_client(credentials={})

#     def test_create_client_with_api_key(self):
#         client = ClaudeProvider.create_client(credentials={"api_key": "sk-test"})
#         assert isinstance(client, anthropic.Anthropic)

#     def test_execute_success_text_input(
#         self, provider, mock_client, text_config, query_params
#     ):
#         mock_client.messages.create.return_value = mock_claude_message(
#             text="ok", model="claude-opus-4-7"
#         )

#         result, error = provider.execute(text_config, query_params, "hi")

#         assert error is None
#         assert result.response.output.content.value == "ok"
#         assert result.response.provider == "anthropic-native"
#         assert result.response.model == "claude-opus-4-7"
#         assert result.response.provider_response_id == "msg_123"
#         assert result.usage.input_tokens == 10
#         assert result.usage.output_tokens == 5
#         assert result.usage.total_tokens == 15

#         call_kwargs = mock_client.messages.create.call_args.kwargs
#         assert call_kwargs["model"] == "claude-opus-4-7"
#         assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
#         assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]

#     def test_execute_does_not_override_user_max_tokens(
#         self, provider, mock_client, query_params
#     ):
#         config = NativeCompletionConfig(
#             provider="anthropic-native",
#             type="text",
#             params={"model": "claude-opus-4-7", "max_tokens": 64},
#         )
#         mock_client.messages.create.return_value = mock_claude_message()

#         provider.execute(config, query_params, "hi")

#         assert mock_client.messages.create.call_args.kwargs["max_tokens"] == 64

#     def test_execute_instructions_renamed_to_system(
#         self, provider, mock_client, query_params
#     ):
#         config = NativeCompletionConfig(
#             provider="anthropic-native",
#             type="text",
#             params={"model": "claude-opus-4-7", "instructions": "be brief"},
#         )
#         mock_client.messages.create.return_value = mock_claude_message()

#         provider.execute(config, query_params, "hi")

#         kwargs = mock_client.messages.create.call_args.kwargs
#         assert kwargs.get("system") == "be brief"
#         assert "instructions" not in kwargs

#     def test_execute_strips_instructions_when_system_also_set(
#         self, provider, mock_client, query_params
#     ):
#         config = NativeCompletionConfig(
#             provider="anthropic-native",
#             type="text",
#             params={
#                 "model": "claude-opus-4-7",
#                 "instructions": "ignored",
#                 "system": "winner",
#             },
#         )
#         mock_client.messages.create.return_value = mock_claude_message()

#         provider.execute(config, query_params, "hi")

#         kwargs = mock_client.messages.create.call_args.kwargs
#         assert kwargs["system"] == "winner"
#         assert "instructions" not in kwargs

#     def test_execute_multimodal_text_image_pdf(
#         self, provider, mock_client, text_config, query_params
#     ):
#         mock_client.messages.create.return_value = mock_claude_message()
#         multimodal = MultiModalInput(
#             parts=[
#                 TextContent(value="describe"),
#                 ImageContent(format="base64", mime_type="image/png", value="ZmFrZQ=="),
#                 PDFContent(
#                     format="url", mime_type="application/pdf", value="https://x/y.pdf"
#                 ),
#             ]
#         )

#         provider.execute(text_config, query_params, multimodal)

#         content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
#         assert content[0] == {"type": "text", "text": "describe"}
#         assert content[1] == {
#             "type": "image",
#             "source": {
#                 "type": "base64",
#                 "media_type": "image/png",
#                 "data": "ZmFrZQ==",
#             },
#         }
#         assert content[2] == {
#             "type": "document",
#             "source": {"type": "url", "url": "https://x/y.pdf"},
#         }

#     def test_execute_strips_conversation_param(
#         self, provider, mock_client, query_params
#     ):
#         config = NativeCompletionConfig(
#             provider="anthropic-native",
#             type="text",
#             params={"model": "claude-opus-4-7", "conversation": {"id": "conv_x"}},
#         )
#         mock_client.messages.create.return_value = mock_claude_message()

#         provider.execute(config, query_params, "hi")

#         assert "conversation" not in mock_client.messages.create.call_args.kwargs

#     def test_execute_joins_only_text_blocks(
#         self, provider, mock_client, text_config, query_params
#     ):
#         # Response with a tool_use block mixed in; we only join text blocks
#         tool_block = SimpleNamespace(type="tool_use", id="t1", name="x", input={})
#         mock_client.messages.create.return_value = mock_claude_message(
#             text="part1",
#             extra_blocks=[tool_block, SimpleNamespace(type="text", text="part2")],
#         )

#         result, error = provider.execute(text_config, query_params, "hi")

#         assert error is None
#         assert result.response.output.content.value == "part1part2"

#     def test_execute_includes_raw_response_when_requested(
#         self, provider, mock_client, text_config, query_params
#     ):
#         mock_client.messages.create.return_value = mock_claude_message()

#         result, _ = provider.execute(
#             text_config, query_params, "hi", include_provider_raw_response=True
#         )

#         assert result.provider_raw_response == {
#             "id": "msg_123",
#             "model": "claude-opus-4-7",
#         }

#     def test_execute_returns_error_on_anthropic_api_error(
#         self, provider, mock_client, text_config, query_params
#     ):
#         mock_client.messages.create.side_effect = anthropic.AnthropicError("boom")

#         result, error = provider.execute(text_config, query_params, "hi")

#         assert result is None
#         assert error is not None
#         assert "boom" in error

#     def test_execute_returns_error_on_unexpected_kwarg(
#         self, provider, mock_client, text_config, query_params
#     ):
#         mock_client.messages.create.side_effect = TypeError(
#             "unexpected keyword argument 'foo'"
#         )

#         result, error = provider.execute(text_config, query_params, "hi")

#         assert result is None
#         assert "Invalid or unexpected parameter" in error
