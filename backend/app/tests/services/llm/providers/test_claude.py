"""Tests for the Anthropic Claude provider.

Covers credential setup, multimodal request shape (text/image/PDF), the
Files API upload path for inline base64 documents, default model / max
tokens behaviour, conversation-key stripping, raw-response passthrough,
and error mapping.
"""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from app.models.llm import (
    ImageContent,
    NativeCompletionConfig,
    PDFContent,
    QueryParams,
    TextContent,
)
from app.services.llm.providers.base import MultiModalInput
from app.services.llm.providers.claude import (
    FILES_API_BETA,
    ClaudeProvider,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mock_message(
    *,
    msg_id: str = "msg_test",
    model: str = "claude-sonnet-4-6",
    text: str = "hello world",
    input_tokens: int = 12,
    output_tokens: int = 7,
) -> MagicMock:
    """Build a stand-in for ``anthropic.types.Message``.

    Uses MagicMock so ``model_dump()`` is callable. Content blocks are
    SimpleNamespace so the provider's ``block.type`` / ``block.text``
    access pattern works as it would with the real SDK objects.
    """
    msg = MagicMock()
    msg.id = msg_id
    msg.model = model
    msg.content = [SimpleNamespace(type="text", text=text)]
    msg.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    msg.model_dump.return_value = {
        "id": msg_id,
        "model": model,
        "content": [{"type": "text", "text": text}],
    }
    return msg


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.fixture
def query() -> QueryParams:
    return QueryParams(input="ignored")


@pytest.fixture
def config() -> NativeCompletionConfig:
    return NativeCompletionConfig(
        provider="anthropic-native",
        type="text",
        params={"model": "claude-sonnet-4-6", "max_tokens": 512},
    )


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _mock_message()
    client.beta.messages.create.return_value = _mock_message(text="beta path ok")
    upload = MagicMock()
    upload.id = "file_abc123"
    client.beta.files.upload.return_value = upload
    return client


@pytest.fixture
def provider(mock_client: MagicMock) -> ClaudeProvider:
    return ClaudeProvider(client=mock_client)


# ---------------------------------------------------------------------------
# create_client
# ---------------------------------------------------------------------------
class TestCreateClient:
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="Anthropic credentials not configured"):
            ClaudeProvider.create_client({})

    def test_returns_anthropic_client(self):
        with patch("app.services.llm.providers.claude.Anthropic") as mock_anthropic:
            mock_anthropic.return_value = MagicMock(name="anthropic-client")
            client = ClaudeProvider.create_client({"api_key": "sk-test"})
        mock_anthropic.assert_called_once_with(api_key="sk-test")
        assert client is mock_anthropic.return_value


# ---------------------------------------------------------------------------
# format_parts — verifies the request shape Anthropic expects
# ---------------------------------------------------------------------------
class TestFormatParts:
    def test_text_block(self):
        out = ClaudeProvider.format_parts([TextContent(value="hi")])
        assert out == [{"type": "text", "text": "hi"}]

    def test_base64_image(self):
        out = ClaudeProvider.format_parts(
            [ImageContent(format="base64", value="b64img", mime_type="image/png")]
        )
        assert out == [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "b64img",
                },
            }
        ]

    def test_url_image(self):
        out = ClaudeProvider.format_parts(
            [
                ImageContent(
                    format="url",
                    value="https://example.com/a.jpg",
                    mime_type="image/jpeg",
                )
            ]
        )
        assert out == [
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/a.jpg"},
            }
        ]

    def test_base64_pdf(self):
        out = ClaudeProvider.format_parts(
            [PDFContent(format="base64", value="b64pdf", mime_type="application/pdf")]
        )
        assert out == [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "b64pdf",
                },
            }
        ]

    def test_url_pdf(self):
        out = ClaudeProvider.format_parts(
            [
                PDFContent(
                    format="url",
                    value="https://example.com/x.pdf",
                    mime_type="application/pdf",
                )
            ]
        )
        assert out == [
            {
                "type": "document",
                "source": {"type": "url", "url": "https://example.com/x.pdf"},
            }
        ]

    def test_mixed_order_preserved(self):
        out = ClaudeProvider.format_parts(
            [
                TextContent(value="describe"),
                ImageContent(format="base64", value="img", mime_type="image/png"),
            ]
        )
        assert [item["type"] for item in out] == ["text", "image"]


# ---------------------------------------------------------------------------
# execute — text-only happy path & defaults
# ---------------------------------------------------------------------------
class TestExecuteText:
    def test_simple_string_input(self, provider, mock_client, config, query):
        resp, err = provider.execute(config, query, "hello")

        assert err is None
        assert resp.response.provider_response_id == "msg_test"
        assert resp.response.model == "claude-sonnet-4-6"
        assert resp.response.output.content.value == "hello world"
        assert resp.usage.input_tokens == 12
        assert resp.usage.output_tokens == 7
        assert resp.usage.total_tokens == 19

        # Non-beta path: plain messages.create was used.
        mock_client.messages.create.assert_called_once()
        mock_client.beta.messages.create.assert_not_called()
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["max_tokens"] == 512
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]

    def test_defaults_model_and_max_tokens_when_missing(
        self, provider, mock_client, query
    ):
        """Empty params → provider falls back to project defaults."""
        cfg = NativeCompletionConfig(
            provider="anthropic-native", type="text", params={}
        )
        resp, err = provider.execute(cfg, query, "hello")

        assert err is None
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["max_tokens"] == 4096

    def test_strips_conversation_key(self, provider, mock_client, query):
        """`conversation` is a Kaapi-level concept; it must not be forwarded
        to the Anthropic SDK (which would raise TypeError)."""
        cfg = NativeCompletionConfig(
            provider="anthropic-native",
            type="text",
            params={"model": "claude-sonnet-4-6", "conversation": "conv_123"},
        )
        resp, err = provider.execute(cfg, query, "hello")

        assert err is None
        kwargs = mock_client.messages.create.call_args.kwargs
        assert "conversation" not in kwargs

    def test_concatenates_multi_block_text_output(
        self, provider, mock_client, config, query
    ):
        mock_client.messages.create.return_value = MagicMock(
            id="msg_multi",
            model="claude-sonnet-4-6",
            content=[
                SimpleNamespace(type="text", text="hello "),
                SimpleNamespace(type="tool_use", text=None),
                SimpleNamespace(type="text", text="world"),
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=2),
        )
        resp, err = provider.execute(config, query, "hi")
        assert err is None
        assert resp.response.output.content.value == "hello world"

    def test_raw_response_included_when_requested(
        self, provider, mock_client, config, query
    ):
        resp, _ = provider.execute(
            config, query, "hello", include_provider_raw_response=True
        )
        assert resp.provider_raw_response == {
            "id": "msg_test",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello world"}],
        }

    def test_raw_response_omitted_by_default(self, provider, config, query):
        resp, _ = provider.execute(config, query, "hello")
        assert resp.provider_raw_response is None


# ---------------------------------------------------------------------------
# execute — multimodal inputs (list of parts and MultiModalInput)
# ---------------------------------------------------------------------------
class TestExecuteMultimodal:
    def test_list_of_parts_forwarded_as_content_blocks(
        self, provider, mock_client, config, query
    ):
        parts = [
            TextContent(value="describe"),
            ImageContent(
                format="url",
                value="https://example.com/cat.jpg",
                mime_type="image/jpeg",
            ),
        ]
        resp, err = provider.execute(config, query, parts)

        assert err is None
        # URL image → no upload, no beta header
        mock_client.beta.files.upload.assert_not_called()
        mock_client.messages.create.assert_called_once()
        kwargs = mock_client.messages.create.call_args.kwargs
        content = kwargs["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "url"

    def test_multimodal_input_wrapper_unwrapped(
        self, provider, mock_client, config, query
    ):
        mm = MultiModalInput(parts=[TextContent(value="hi")])
        resp, err = provider.execute(config, query, mm)

        assert err is None
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["messages"][0]["content"] == [{"type": "text", "text": "hi"}]


# ---------------------------------------------------------------------------
# execute — Files API upload path for base64 documents/images
# ---------------------------------------------------------------------------
class TestFilesApiUploadPath:
    def test_base64_pdf_uploaded_and_referenced_by_file_id(
        self, provider, mock_client, config, query
    ):
        pdf_bytes = b"%PDF-1.4 fake"
        parts = [
            TextContent(value="summarize"),
            PDFContent(
                format="base64", value=_b64(pdf_bytes), mime_type="application/pdf"
            ),
        ]
        resp, err = provider.execute(config, query, parts)

        assert err is None
        # Beta endpoint used because we uploaded a file
        mock_client.beta.messages.create.assert_called_once()
        mock_client.messages.create.assert_not_called()

        # File was uploaded with decoded bytes + correct media type
        upload_kwargs = mock_client.beta.files.upload.call_args.kwargs
        filename, file_obj, media_type = upload_kwargs["file"]
        assert filename == "document.pdf"
        assert media_type == "application/pdf"
        assert file_obj.read() == pdf_bytes

        # Block was rewritten to reference the uploaded file_id
        beta_kwargs = mock_client.beta.messages.create.call_args.kwargs
        pdf_block = beta_kwargs["messages"][0]["content"][1]
        assert pdf_block["type"] == "document"
        assert pdf_block["source"] == {"type": "file", "file_id": "file_abc123"}

        # Beta header is appended without dropping any existing values
        assert FILES_API_BETA in beta_kwargs["betas"]

    def test_base64_image_uploaded_via_files_api(
        self, provider, mock_client, config, query
    ):
        img_bytes = b"\x89PNG\r\n\x1a\nfake"
        parts = [
            ImageContent(format="base64", value=_b64(img_bytes), mime_type="image/png")
        ]
        resp, err = provider.execute(config, query, parts)

        assert err is None
        mock_client.beta.files.upload.assert_called_once()
        upload_kwargs = mock_client.beta.files.upload.call_args.kwargs
        filename, _, media_type = upload_kwargs["file"]
        assert filename == "image"
        assert media_type == "image/png"

        beta_kwargs = mock_client.beta.messages.create.call_args.kwargs
        block = beta_kwargs["messages"][0]["content"][0]
        assert block["source"] == {"type": "file", "file_id": "file_abc123"}

    def test_existing_betas_preserved(self, provider, mock_client, query):
        """Caller-supplied beta headers must not be clobbered."""
        cfg = NativeCompletionConfig(
            provider="anthropic-native",
            type="text",
            params={
                "model": "claude-sonnet-4-6",
                "max_tokens": 512,
                "betas": ["caller-beta-1"],
            },
        )
        parts = [
            PDFContent(format="base64", value=_b64(b"pdf"), mime_type="application/pdf")
        ]
        resp, err = provider.execute(cfg, query, parts)

        assert err is None
        beta_kwargs = mock_client.beta.messages.create.call_args.kwargs
        assert beta_kwargs["betas"] == ["caller-beta-1", FILES_API_BETA]

    def test_url_pdf_does_not_trigger_upload(
        self, provider, mock_client, config, query
    ):
        parts = [
            PDFContent(
                format="url",
                value="https://example.com/doc.pdf",
                mime_type="application/pdf",
            )
        ]
        resp, err = provider.execute(config, query, parts)

        assert err is None
        mock_client.beta.files.upload.assert_not_called()
        mock_client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# execute — error mapping
# ---------------------------------------------------------------------------
class TestExecuteErrors:
    def test_type_error_returns_clean_message(
        self, provider, mock_client, config, query
    ):
        mock_client.messages.create.side_effect = TypeError(
            "unexpected keyword argument 'nonsense'"
        )
        resp, err = provider.execute(config, query, "hi")
        assert resp is None
        assert "Invalid or unexpected parameter in Config" in err
        assert "nonsense" in err

    def test_anthropic_error_returns_clean_message(
        self, provider, mock_client, config, query
    ):
        mock_client.messages.create.side_effect = anthropic.AnthropicError(
            "rate limited"
        )
        resp, err = provider.execute(config, query, "hi")
        assert resp is None
        assert "Anthropic API error" in err
        assert "rate limited" in err

    def test_generic_exception_returns_opaque_message(
        self, provider, mock_client, config, query
    ):
        """Unexpected errors are logged but the surface message must not leak
        internals to the caller."""
        mock_client.messages.create.side_effect = RuntimeError("boom internal detail")
        resp, err = provider.execute(config, query, "hi")
        assert resp is None
        assert err == "Unexpected error occurred"
