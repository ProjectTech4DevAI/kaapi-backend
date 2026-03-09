import pytest
from unittest.mock import MagicMock

from app.models.llm.request import (
    TextInput,
    AudioInput,
    ImageInput,
    PDFInput,
    TextContent,
    AudioContent,
    ImageContent,
    PDFContent,
    NativeCompletionConfig,
    QueryParams,
)
from app.services.llm.providers.base import (
    ContentPart,
    MultiModalInput,
)
from app.services.llm.providers.oai import OpenAIProvider
from app.services.llm.providers.gai import GoogleAIProvider
from app.utils import (
    resolve_input,
    resolve_image_content,
    resolve_pdf_content,
)


class TestMultiModalInput:
    def test_valid_parts(self):
        mm = MultiModalInput(
            parts=[
                TextContent(value="hello"),
                ImageContent(format="base64", value="abc", mime_type="image/png"),
                PDFContent(format="base64", value="abc", mime_type="application/pdf"),
            ]
        )
        assert len(mm.parts) == 3

    def test_empty_parts_raises(self):
        with pytest.raises(Exception):
            MultiModalInput(parts=[])

    def test_single_text_part(self):
        mm = MultiModalInput(parts=[TextContent(value="only text")])
        assert len(mm.parts) == 1


class TestResolveInputMultimodal:
    def test_image_input_returns_image_content_list(self):
        img = ImageInput(
            content=ImageContent(format="base64", value="abc", mime_type="image/png")
        )
        result, error = resolve_input(img)
        assert error is None
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ImageContent)

    def test_pdf_input_returns_pdf_content_list(self):
        pdf = PDFInput(
            content=PDFContent(
                format="base64", value="abc", mime_type="application/pdf"
            )
        )
        result, error = resolve_input(pdf)
        assert error is None
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], PDFContent)

    def test_multimodal_list_returns_multimodal_input(self):
        inputs = [
            TextInput(content=TextContent(value="describe")),
            ImageInput(
                content=ImageContent(
                    format="base64", value="abc", mime_type="image/png"
                )
            ),
        ]
        result, error = resolve_input(inputs)
        assert error is None
        assert isinstance(result, MultiModalInput)
        assert len(result.parts) == 2

    def test_multimodal_list_with_pdf(self):
        inputs = [
            TextInput(content=TextContent(value="analyze")),
            PDFInput(
                content=PDFContent(
                    format="base64", value="abc", mime_type="application/pdf"
                )
            ),
        ]
        result, error = resolve_input(inputs)
        assert error is None
        assert isinstance(result, MultiModalInput)
        assert len(result.parts) == 2

    def test_multimodal_list_with_audio_rejected(self):
        inputs = [
            TextInput(content=TextContent(value="hello")),
            AudioInput(content=AudioContent(value="abc", mime_type="audio/wav")),
        ]
        result, error = resolve_input(inputs)
        assert error is not None
        assert "audio" in error.lower()
        assert "stt" in error.lower()

    def test_image_input_default_mime_type(self):
        img = ImageInput(content=ImageContent(format="base64", value="abc"))
        result, error = resolve_input(img)
        assert error is None
        assert result[0].mime_type == "image/png"

    def test_pdf_input_default_mime_type(self):
        pdf = PDFInput(content=PDFContent(format="base64", value="abc"))
        result, error = resolve_input(pdf)
        assert error is None
        assert result[0].mime_type == "application/pdf"

    def test_image_input_multiple_contents(self):
        img = ImageInput(
            content=[
                ImageContent(format="base64", value="abc1", mime_type="image/png"),
                ImageContent(
                    format="url",
                    value="https://example.com/img.jpg",
                    mime_type="image/jpeg",
                ),
            ]
        )
        result, error = resolve_input(img)
        assert error is None
        assert len(result) == 2

    def test_multimodal_mixed_types_in_parts(self):
        inputs = [
            TextInput(content=TextContent(value="look at these")),
            ImageInput(
                content=ImageContent(
                    format="base64", value="img", mime_type="image/png"
                )
            ),
            PDFInput(
                content=PDFContent(
                    format="base64", value="pdf", mime_type="application/pdf"
                )
            ),
        ]
        result, error = resolve_input(inputs)
        assert error is None
        assert isinstance(result, MultiModalInput)
        assert len(result.parts) == 3
        assert isinstance(result.parts[0], TextContent)
        assert isinstance(result.parts[1], ImageContent)
        assert isinstance(result.parts[2], PDFContent)


class TestOpenAIFormatParts:
    def test_text_part(self):
        parts = [TextContent(value="hello")]
        result = OpenAIProvider.format_parts(parts)
        assert result == [{"type": "input_text", "text": "hello"}]

    def test_image_base64_part(self):
        parts = [ImageContent(format="base64", value="abc123", mime_type="image/png")]
        result = OpenAIProvider.format_parts(parts)
        assert len(result) == 1
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"] == "data:image/png;base64,abc123"

    def test_image_url_part(self):
        parts = [
            ImageContent(
                format="url",
                value="https://example.com/img.jpg",
                mime_type="image/jpeg",
            )
        ]
        result = OpenAIProvider.format_parts(parts)
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"] == "https://example.com/img.jpg"

    def test_pdf_base64_part(self):
        parts = [
            PDFContent(format="base64", value="pdf123", mime_type="application/pdf")
        ]
        result = OpenAIProvider.format_parts(parts)
        assert len(result) == 1
        assert result[0]["type"] == "input_file"
        assert result[0]["file_url"] == "data:application/pdf;base64,pdf123"

    def test_pdf_url_part(self):
        parts = [
            PDFContent(
                format="url",
                value="https://example.com/doc.pdf",
                mime_type="application/pdf",
            )
        ]
        result = OpenAIProvider.format_parts(parts)
        assert result[0]["type"] == "input_file"
        assert result[0]["file_url"] == "https://example.com/doc.pdf"

    def test_mixed_parts(self):
        parts = [
            TextContent(value="describe"),
            ImageContent(format="base64", value="img", mime_type="image/png"),
            PDFContent(
                format="url",
                value="https://example.com/doc.pdf",
                mime_type="application/pdf",
            ),
        ]
        result = OpenAIProvider.format_parts(parts)
        assert len(result) == 3
        assert result[0]["type"] == "input_text"
        assert result[1]["type"] == "input_image"
        assert result[2]["type"] == "input_file"


class TestGoogleAIFormatParts:
    def test_text_part(self):
        parts = [TextContent(value="hello")]
        result = GoogleAIProvider.format_parts(parts)
        assert result == [{"text": "hello"}]

    def test_image_base64_part(self):
        parts = [ImageContent(format="base64", value="abc123", mime_type="image/png")]
        result = GoogleAIProvider.format_parts(parts)
        assert len(result) == 1
        assert result[0] == {
            "inline_data": {"data": "abc123", "mime_type": "image/png"}
        }

    def test_image_url_part(self):
        parts = [
            ImageContent(
                format="url",
                value="https://example.com/img.jpg",
                mime_type="image/jpeg",
            )
        ]
        result = GoogleAIProvider.format_parts(parts)
        assert result[0] == {
            "file_data": {
                "file_uri": "https://example.com/img.jpg",
                "mime_type": "image/jpeg",
                "display_name": None,
            }
        }

    def test_pdf_base64_part(self):
        parts = [
            PDFContent(format="base64", value="pdf123", mime_type="application/pdf")
        ]
        result = GoogleAIProvider.format_parts(parts)
        assert result[0] == {
            "inline_data": {"data": "pdf123", "mime_type": "application/pdf"}
        }

    def test_pdf_url_part(self):
        parts = [
            PDFContent(
                format="url",
                value="https://example.com/doc.pdf",
                mime_type="application/pdf",
            )
        ]
        result = GoogleAIProvider.format_parts(parts)
        assert result[0] == {
            "file_data": {
                "file_uri": "https://example.com/doc.pdf",
                "mime_type": "application/pdf",
                "display_name": None,
            }
        }

    def test_mixed_parts(self):
        parts = [
            TextContent(value="analyze"),
            ImageContent(
                format="url", value="https://img.com/a.jpg", mime_type="image/jpeg"
            ),
            PDFContent(format="base64", value="pdf", mime_type="application/pdf"),
        ]
        result = GoogleAIProvider.format_parts(parts)
        assert len(result) == 3
        assert "text" in result[0]
        assert "file_data" in result[1]
        assert "inline_data" in result[2]


class TestResolveImageContent:
    def test_single_content(self):
        img = ImageInput(
            content=ImageContent(format="base64", value="abc", mime_type="image/png")
        )
        result = resolve_image_content(img)
        assert len(result) == 1
        assert result[0].mime_type == "image/png"

    def test_default_mime_type(self):
        img = ImageInput(content=ImageContent(format="base64", value="abc"))
        result = resolve_image_content(img)
        assert result[0].mime_type == "image/png"

    def test_list_content(self):
        img = ImageInput(
            content=[
                ImageContent(format="base64", value="a", mime_type="image/png"),
                ImageContent(format="base64", value="b", mime_type="image/jpeg"),
            ]
        )
        result = resolve_image_content(img)
        assert len(result) == 2


class TestResolvePdfContent:
    def test_single_content(self):
        pdf = PDFInput(
            content=PDFContent(
                format="base64", value="abc", mime_type="application/pdf"
            )
        )
        result = resolve_pdf_content(pdf)
        assert len(result) == 1
        assert result[0].mime_type == "application/pdf"

    def test_default_mime_type(self):
        pdf = PDFInput(content=PDFContent(format="base64", value="abc"))
        result = resolve_pdf_content(pdf)
        assert result[0].mime_type == "application/pdf"

    def test_list_content(self):
        pdf = PDFInput(
            content=[
                PDFContent(format="base64", value="a", mime_type="application/pdf"),
                PDFContent(
                    format="url",
                    value="https://example.com/doc.pdf",
                    mime_type="application/pdf",
                ),
            ]
        )
        result = resolve_pdf_content(pdf)
        assert len(result) == 2


class TestResolveInputEdgeCases:
    def test_unknown_input_type(self):
        result, error = resolve_input(12345)
        assert error is not None
        assert "Unknown input type" in error

    def test_unsupported_type_in_multimodal_list(self):
        result, error = resolve_input(["not_a_valid_input"])
        assert error is not None
        assert "Unsupported input type" in error

    def test_text_input_resolves_string(self):
        text = TextInput(content=TextContent(value="hello world"))
        result, error = resolve_input(text)
        assert error is None
        assert result == "hello world"


class TestOpenAIExecuteInputRouting:
    def _make_provider(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.id = "resp_123"
        mock_resp.model = "gpt-4o-mini"
        mock_resp.output_text = "result"
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5
        mock_resp.usage.total_tokens = 15
        mock_resp.conversation = None
        mock_client.responses.create.return_value = mock_resp
        return OpenAIProvider(client=mock_client), mock_client

    def _make_config(self):
        return NativeCompletionConfig(
            provider="openai-native", type="text", params={"model": "gpt-4o-mini"}
        )

    def _make_query(self):
        return QueryParams(input="test")

    def test_multimodal_input(self):
        provider, mock_client = self._make_provider()
        mm = MultiModalInput(
            parts=[
                TextContent(value="describe"),
                ImageContent(format="base64", value="img", mime_type="image/png"),
            ]
        )
        response, error = provider.execute(
            completion_config=self._make_config(),
            query=self._make_query(),
            resolved_input=mm,
        )
        assert error is None
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["input"][0]["role"] == "user"
        assert len(call_kwargs["input"][0]["content"]) == 2

    def test_list_input(self):
        provider, mock_client = self._make_provider()
        parts = [ImageContent(format="base64", value="img", mime_type="image/png")]
        response, error = provider.execute(
            completion_config=self._make_config(),
            query=self._make_query(),
            resolved_input=parts,
        )
        assert error is None
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["input"][0]["role"] == "user"

    def test_string_input(self):
        provider, mock_client = self._make_provider()
        response, error = provider.execute(
            completion_config=self._make_config(),
            query=self._make_query(),
            resolved_input="hello",
        )
        assert error is None
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["input"] == "hello"


class TestGoogleAIExecuteTextRouting:
    def _make_provider(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.response_id = "resp_gai_123"
        mock_resp.model_version = "gemini-2.0-flash"
        mock_resp.text = "response text"
        mock_resp.usage_metadata.prompt_token_count = 10
        mock_resp.usage_metadata.candidates_token_count = 5
        mock_resp.usage_metadata.total_token_count = 15
        mock_resp.usage_metadata.thoughts_token_count = 0
        mock_client.models.generate_content.return_value = mock_resp
        return GoogleAIProvider(client=mock_client), mock_client

    def _make_config(self, **extra_params):
        params = {"model": "gemini-2.0-flash"}
        params.update(extra_params)
        return NativeCompletionConfig(
            provider="google-native", type="text", params=params
        )

    def _make_query(self):
        return QueryParams(input="test")

    def test_multimodal_input(self):
        provider, mock_client = self._make_provider()
        mm = MultiModalInput(
            parts=[
                TextContent(value="describe"),
                ImageContent(format="base64", value="img", mime_type="image/png"),
            ]
        )
        response, error = provider.execute(
            completion_config=self._make_config(),
            query=self._make_query(),
            resolved_input=mm,
        )
        assert error is None
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["contents"][0]["role"] == "user"
        assert len(call_kwargs["contents"][0]["parts"]) == 2

    def test_list_input(self):
        provider, mock_client = self._make_provider()
        parts = [ImageContent(format="base64", value="img", mime_type="image/png")]
        response, error = provider.execute(
            completion_config=self._make_config(),
            query=self._make_query(),
            resolved_input=parts,
        )
        assert error is None
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["contents"][0]["role"] == "user"

    def test_string_input(self):
        provider, mock_client = self._make_provider()
        response, error = provider.execute(
            completion_config=self._make_config(),
            query=self._make_query(),
            resolved_input="hello",
        )
        assert error is None
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["contents"][0]["parts"] == [{"text": "hello"}]

    def test_missing_model(self):
        provider, _ = self._make_provider()
        config = NativeCompletionConfig(
            provider="google-native", type="text", params={}
        )
        response, error = provider.execute(
            completion_config=config,
            query=self._make_query(),
            resolved_input="hello",
        )
        assert response is None
        assert "Missing 'model'" in error

    def test_instructions_passed_to_config(self):
        provider, mock_client = self._make_provider()
        response, error = provider.execute(
            completion_config=self._make_config(instructions="be helpful"),
            query=self._make_query(),
            resolved_input="hello",
        )
        assert error is None
        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert config.system_instruction == "be helpful"

    def test_no_usage_metadata(self):
        provider, mock_client = self._make_provider()
        mock_resp = mock_client.models.generate_content.return_value
        mock_resp.usage_metadata = None
        response, error = provider.execute(
            completion_config=self._make_config(),
            query=self._make_query(),
            resolved_input="hello",
        )
        assert error is None
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0
