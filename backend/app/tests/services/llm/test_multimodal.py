import pytest

from app.models.llm.request import (
    TextInput,
    AudioInput,
    ImageInput,
    PDFInput,
    TextContent,
    AudioContent,
    ImageContent,
    PDFContent,
)
from app.services.llm.providers.base import (
    ContentPart,
    MultiModalInput,
    validate_completion_input,
    _get_content_label,
)
from app.services.llm.providers.oai import OpenAIProvider
from app.services.llm.providers.gai import GoogleAIProvider
from app.utils import (
    resolve_input,
    resolve_image_content,
    resolve_pdf_content,
)


class TestValidateCompletionInput:
    def test_text_with_str_passes(self):
        assert validate_completion_input("text", "hello") is None

    def test_stt_with_str_passes(self):
        assert validate_completion_input("stt", "/tmp/audio.wav") is None

    def test_tts_with_str_passes(self):
        assert validate_completion_input("tts", "say this") is None

    def test_image_with_image_content_list_passes(self):
        parts = [ImageContent(format="base64", value="abc", mime_type="image/png")]
        assert validate_completion_input("image", parts) is None

    def test_pdf_with_pdf_content_list_passes(self):
        parts = [PDFContent(format="base64", value="abc", mime_type="application/pdf")]
        assert validate_completion_input("pdf", parts) is None

    def test_multimodal_with_multimodal_input_passes(self):
        mm = MultiModalInput(
            parts=[
                TextContent(value="hello"),
                ImageContent(format="base64", value="abc", mime_type="image/png"),
            ]
        )
        assert validate_completion_input("multimodal", mm) is None

    def test_text_input_with_pdf_completion_fails(self):
        error = validate_completion_input("pdf", "some text")
        assert error is not None
        assert "input type mismatch" in error.lower()
        assert "'pdf'" in error
        assert "text" in error

    def test_multimodal_input_with_image_completion_fails(self):
        mm = MultiModalInput(
            parts=[
                TextContent(value="hello"),
                ImageContent(format="base64", value="abc", mime_type="image/png"),
            ]
        )
        error = validate_completion_input("image", mm)
        assert error is not None
        assert "multimodal" in error.lower()
        assert "set completion type to 'multimodal'" in error

    def test_text_input_with_image_completion_no_multimodal_hint(self):
        error = validate_completion_input("image", "some text")
        assert error is not None
        assert "set completion type to 'multimodal'" not in error
        assert "Please ensure the input type matches" in error

    def test_pdf_content_in_image_completion_fails(self):
        parts = [PDFContent(format="base64", value="abc", mime_type="application/pdf")]
        error = validate_completion_input("image", parts)
        assert error is not None
        assert "'pdf'" in error

    def test_image_content_in_pdf_completion_fails(self):
        parts = [ImageContent(format="base64", value="abc", mime_type="image/png")]
        error = validate_completion_input("pdf", parts)
        assert error is not None
        assert "'image'" in error

    def test_unknown_completion_type(self):
        error = validate_completion_input("unknown_type", "hello")
        assert error is not None
        assert "Unknown completion type" in error

    def test_list_input_with_text_completion_fails(self):
        parts = [ImageContent(format="base64", value="abc", mime_type="image/png")]
        error = validate_completion_input("text", parts)
        assert error is not None
        assert "text" in error


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


class TestGetContentLabel:
    def test_text_content(self):
        assert _get_content_label(TextContent(value="hi")) == "text"

    def test_image_content(self):
        assert (
            _get_content_label(
                ImageContent(format="base64", value="abc", mime_type="image/png")
            )
            == "image"
        )

    def test_pdf_content(self):
        assert (
            _get_content_label(
                PDFContent(format="base64", value="abc", mime_type="application/pdf")
            )
            == "pdf"
        )

    def test_audio_content(self):
        assert (
            _get_content_label(AudioContent(value="abc", mime_type="audio/wav"))
            == "audio"
        )


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
