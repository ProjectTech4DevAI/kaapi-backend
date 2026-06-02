"""
Unit tests for app/utils.py — functions not already covered by
test_input_resolver.py and test_callback_ssrf.py.
"""

import base64
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import openai
import pytest
import requests

from app.utils import (
    APIResponse,
    ValidationErrorDetail,
    download_audio_bytes,
    generate_eval_completion_email,
    handle_openai_error,
    mask_string,
    require_organization_for_project,
    resolve_audio_url,
    resolve_image_content,
    resolve_input,
    resolve_pdf_content,
    send_callback,
    cleanup_temp_file,
    MAX_AUDIO_SIZE,
)
from app.models.llm.request import (
    AudioContent,
    AudioInput,
    ImageContent,
    ImageInput,
    PDFContent,
    PDFInput,
    TextContent,
    TextInput,
)


# ---------------------------------------------------------------------------
# APIResponse
# ---------------------------------------------------------------------------
class TestAPIResponse:
    def test_success_response(self) -> None:
        resp = APIResponse.success_response(data={"id": 1})
        assert resp.success is True
        assert resp.data == {"id": 1}
        assert resp.error is None

    def test_success_response_with_metadata(self) -> None:
        resp = APIResponse.success_response(data="ok", metadata={"page": 1, "total": 5})
        assert resp.metadata == {"page": 1, "total": 5}

    def test_failure_response_string_error(self) -> None:
        resp = APIResponse.failure_response(error="Something broke")
        assert resp.success is False
        assert resp.error == "Something broke"
        assert resp.errors is None

    def test_failure_response_validation_errors(self) -> None:
        raw_errors = [
            {"loc": ("body", "name"), "msg": "Value error, name is required"},
            {"loc": ("body", "age"), "msg": "Type error, not a number"},
            {"loc": (), "msg": "Assertion error, global check failed"},
        ]
        resp = APIResponse.failure_response(error=raw_errors)
        assert resp.success is False
        assert resp.error == "Validation failed"
        assert len(resp.errors) == 3
        assert resp.errors[0].field == "name"
        assert resp.errors[0].message == "name is required"  # prefix stripped
        assert resp.errors[1].message == "not a number"
        assert resp.errors[2].field == "unknown"  # empty loc


# ---------------------------------------------------------------------------
# mask_string
# ---------------------------------------------------------------------------
class TestMaskString:
    def test_empty_string(self) -> None:
        assert mask_string("") == ""

    def test_short_string(self) -> None:
        result = mask_string("ab")
        # length=2, num_mask=1, start=0, end=1 → "*b"
        assert "*" in result
        assert len(result) == 2

    def test_masks_middle_portion(self) -> None:
        result = mask_string("abcdef")
        # length=6, num_mask=3, start=1, end=4 → "a***ef"
        assert result == "a***ef"

    def test_custom_mask_char(self) -> None:
        result = mask_string("abcdef", mask_char="#")
        assert result == "a###ef"


# ---------------------------------------------------------------------------
# handle_openai_error
# ---------------------------------------------------------------------------
class TestHandleOpenAIError:
    def test_extracts_message_from_body_dict(self) -> None:
        err = openai.BadRequestError(
            message="bad", response=MagicMock(), body={"message": "quota exceeded"}
        )
        assert handle_openai_error(err) == "quota exceeded"

    def test_falls_back_to_message_attr(self) -> None:
        err = openai.APIConnectionError(request=MagicMock())
        err.message = "connection timeout"
        err.body = None
        assert handle_openai_error(err) == "connection timeout"

    def test_falls_back_to_response_json(self) -> None:
        resp_mock = MagicMock()
        resp_mock.json.return_value = {"error": {"message": "rate limited"}}
        err = openai.OpenAIError()
        err.body = None
        err.response = resp_mock
        # Remove message so hasattr(e, "message") is False and we hit the response branch
        if hasattr(err, "message"):
            del err.message
        assert handle_openai_error(err) == "rate limited"

    def test_falls_back_to_str(self) -> None:
        err = openai.OpenAIError("generic failure")
        err.body = None
        # Remove both message and response so we fall through to str(e)
        if hasattr(err, "message"):
            del err.message
        if hasattr(err, "response"):
            del err.response
        assert "generic failure" in handle_openai_error(err)


# ---------------------------------------------------------------------------
# require_organization_for_project
# ---------------------------------------------------------------------------
class TestRequireOrganizationForProject:
    def test_raises_when_project_without_org(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            require_organization_for_project(project_id=1, organization_id=None)
        assert exc_info.value.status_code == 400

    def test_passes_when_both_set(self) -> None:
        require_organization_for_project(project_id=1, organization_id=2)

    def test_passes_when_both_none(self) -> None:
        require_organization_for_project(project_id=None, organization_id=None)

    def test_passes_when_only_org(self) -> None:
        require_organization_for_project(project_id=None, organization_id=5)


# ---------------------------------------------------------------------------
# download_audio_bytes
# ---------------------------------------------------------------------------
class TestDownloadAudioBytes:
    @patch("app.utils.validate_callback_url")
    @patch("app.utils.requests.get")
    def test_successful_download(self, mock_get, mock_validate) -> None:
        audio_data = b"fake-audio-bytes"
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "audio/wav", "Content-Length": "16"}
        mock_resp.iter_content.return_value = [audio_data]
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_resp

        data, error = download_audio_bytes("https://cdn.example.com/audio.wav")
        assert error is None
        assert data == audio_data

    @patch("app.utils.validate_callback_url")
    def test_rejects_non_https_url(self, mock_validate) -> None:
        mock_validate.side_effect = ValueError("Only HTTPS URLs are allowed")
        data, error = download_audio_bytes("http://example.com/audio.wav")
        assert data is None
        assert "Invalid public URL" in error

    @patch("app.utils.validate_callback_url")
    @patch("app.utils.requests.get")
    def test_rejects_non_audio_content_type(self, mock_get, mock_validate) -> None:
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_resp

        data, error = download_audio_bytes("https://example.com/page.html")
        assert data is None
        assert "Content-Type" in error

    @patch("app.utils.validate_callback_url")
    @patch("app.utils.requests.get")
    def test_rejects_oversized_content_length(self, mock_get, mock_validate) -> None:
        mock_resp = MagicMock()
        mock_resp.headers = {
            "Content-Type": "audio/wav",
            "Content-Length": str(MAX_AUDIO_SIZE + 1),
        }
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_resp

        data, error = download_audio_bytes("https://example.com/huge.wav")
        assert data is None
        assert "too large" in error.lower() or "File" in error

    @patch("app.utils.validate_callback_url")
    @patch("app.utils.requests.get")
    def test_rejects_oversized_during_streaming(self, mock_get, mock_validate) -> None:
        """Server lies about Content-Length; actual stream exceeds limit."""
        chunk = b"x" * 8192
        # Enough chunks to exceed MAX_AUDIO_SIZE
        num_chunks = (MAX_AUDIO_SIZE // 8192) + 2

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "audio/wav"}
        mock_resp.iter_content.return_value = (chunk for _ in range(num_chunks))
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_resp

        data, error = download_audio_bytes("https://example.com/sneaky.wav")
        assert data is None
        assert "exceeded" in error.lower() or "max size" in error.lower()

    @patch("app.utils.validate_callback_url")
    @patch("app.utils.requests.get")
    def test_handles_timeout(self, mock_get, mock_validate) -> None:
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        data, error = download_audio_bytes("https://slow.example.com/audio.wav")
        assert data is None
        assert "Timed out" in error

    @patch("app.utils.validate_callback_url")
    @patch("app.utils.requests.get")
    def test_handles_http_error(self, mock_get, mock_validate) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_err = requests.exceptions.HTTPError(response=mock_resp)
        mock_get.side_effect = http_err

        data, error = download_audio_bytes("https://example.com/missing.wav")
        assert data is None
        assert "404" in error


# ---------------------------------------------------------------------------
# resolve_audio_url
# ---------------------------------------------------------------------------
class TestResolveAudioUrl:
    @patch("app.utils.download_audio_bytes")
    def test_returns_audio_ref(self, mock_download) -> None:
        from app.core.audio_utils import AudioRef

        audio_data = b"RIFF" + b"\x00" * 36
        mock_download.return_value = (audio_data, None)

        ref, error = resolve_audio_url("https://cdn.example.com/a.wav", "audio/wav")
        assert error is None
        assert isinstance(ref, AudioRef)
        assert ref.bytes_ == audio_data
        assert ref.mime_type == "audio/wav"

    @patch("app.utils.download_audio_bytes")
    def test_propagates_download_error(self, mock_download) -> None:
        mock_download.return_value = (None, "Timed out downloading audio from URL")

        ref, error = resolve_audio_url("https://example.com/a.wav", "audio/wav")
        assert ref is None
        assert "Timed out" in error


# ---------------------------------------------------------------------------
# resolve_image_content / resolve_pdf_content
# ---------------------------------------------------------------------------
class TestResolveImageContent:
    def test_single_image_gets_default_mime(self) -> None:
        img = ImageInput(content=ImageContent(value="abc123", mime_type=None))
        result = resolve_image_content(img)
        assert len(result) == 1
        assert result[0].mime_type == "image/png"

    def test_list_of_images_preserves_existing_mime(self) -> None:
        img = ImageInput(
            content=[
                ImageContent(value="a", mime_type="image/jpeg"),
                ImageContent(value="b", mime_type=None),
            ]
        )
        result = resolve_image_content(img)
        assert len(result) == 2
        assert result[0].mime_type == "image/jpeg"
        assert result[1].mime_type == "image/png"


class TestResolvePdfContent:
    def test_single_pdf_gets_default_mime(self) -> None:
        pdf = PDFInput(content=PDFContent(value="data", mime_type=None))
        result = resolve_pdf_content(pdf)
        assert len(result) == 1
        assert result[0].mime_type == "application/pdf"

    def test_list_of_pdfs(self) -> None:
        pdf = PDFInput(
            content=[
                PDFContent(value="a", mime_type="application/pdf"),
                PDFContent(value="b", mime_type=None),
            ]
        )
        result = resolve_pdf_content(pdf)
        assert len(result) == 2
        assert result[1].mime_type == "application/pdf"


# ---------------------------------------------------------------------------
# resolve_input — image, pdf, multimodal, url-audio, and error paths
# ---------------------------------------------------------------------------
class TestResolveInputExtended:
    def test_image_input(self) -> None:
        img = ImageInput(content=ImageContent(value="b64data", mime_type="image/jpeg"))
        result, error = resolve_input(img)
        assert error is None
        assert len(result) == 1
        assert result[0].mime_type == "image/jpeg"

    def test_pdf_input(self) -> None:
        pdf = PDFInput(content=PDFContent(value="b64pdf", mime_type=None))
        result, error = resolve_input(pdf)
        assert error is None
        assert len(result) == 1
        assert result[0].mime_type == "application/pdf"

    @patch("app.utils.resolve_audio_url")
    def test_audio_url_input(self, mock_resolve_url) -> None:
        from app.core.audio_utils import AudioRef

        mocked_ref = AudioRef(bytes_=b"audio", mime_type="audio/wav")
        mock_resolve_url.return_value = (mocked_ref, None)
        audio = AudioInput(
            content=AudioContent(
                format="url",
                value="https://cdn.example.com/a.wav",
                mime_type="audio/wav",
            )
        )
        result, error = resolve_input(audio)
        assert error is None
        assert result is mocked_ref

    def test_multimodal_text_and_image(self) -> None:
        parts = [
            TextInput(content=TextContent(value="describe this")),
            ImageInput(content=ImageContent(value="b64img", mime_type="image/png")),
        ]
        result, error = resolve_input(parts)
        assert error is None
        assert len(result.parts) == 2

    def test_multimodal_rejects_audio(self) -> None:
        parts = [
            TextInput(content=TextContent(value="transcribe")),
            AudioInput(content=AudioContent(value="b64audio", mime_type="audio/wav")),
        ]
        result, error = resolve_input(parts)
        assert result == ""
        assert "not supported in multimodal" in error

    def test_multimodal_rejects_unknown_type(self) -> None:
        result, error = resolve_input(["not a valid input"])
        # list with unsupported item type
        assert result == ""
        assert "Unsupported input type" in error

    def test_unknown_input_type(self) -> None:
        result, error = resolve_input("just a string")
        assert result == ""
        assert "Unknown input type" in error


# ---------------------------------------------------------------------------
# send_callback with webhook signing
# ---------------------------------------------------------------------------
class TestSendCallbackWithSigning:
    @patch("app.utils.validate_callback_url")
    @patch("requests.Session")
    def test_includes_signature_headers(self, mock_session_cls, mock_validate) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value.__enter__.return_value = mock_session

        result = send_callback(
            "https://api.example.com/hook",
            {"event": "done"},
            webhook_secret="test-secret",
        )
        assert result is True
        headers = mock_session.post.call_args[1]["headers"]
        assert "X-Webhook-Signature" in headers
        assert "X-Webhook-Timestamp" in headers
        assert len(headers["X-Webhook-Signature"]) == 64  # sha256 hex

    @patch("app.utils.validate_callback_url")
    def test_returns_false_for_blocked_url(self, mock_validate) -> None:
        mock_validate.side_effect = ValueError("private IP")
        result = send_callback("https://internal/hook", {"x": 1})
        assert result is False


class TestGenerateEvalCompletionEmail:
    def test_completed_renders_expected_fields(self) -> None:
        data = generate_eval_completion_email(
            run_name="exp-1",
            project_name="Demo",
            status="completed",
            completed_at="May 16, 2026 at 6:33 PM",
            link="https://app.example.com/evaluations/",
            error_message=None,
        )
        assert "Completed" in data.subject
        assert "exp-1" in data.subject
        assert "Completed" in data.html_content
        assert "exp-1" in data.html_content
        assert "Demo" in data.html_content
        assert "May 16, 2026 at 6:33 PM" in data.html_content
        assert "https://app.example.com/evaluations/" in data.html_content

    def test_failed_status_changes_label_and_subject(self) -> None:
        data = generate_eval_completion_email(
            run_name="exp-1",
            project_name="Demo",
            status="failed",
            completed_at="May 16, 2026 at 6:33 PM",
            link="https://app.example.com/evaluations/",
            error_message="Batch failed: timeout",
        )
        assert "Failed" in data.subject
        assert "Failed" in data.html_content
        assert "Batch failed: timeout" in data.html_content

    def test_no_error_omits_error_block(self) -> None:
        data = generate_eval_completion_email(
            run_name="exp-1",
            project_name="Demo",
            status="completed",
            completed_at="May 16, 2026 at 6:33 PM",
            link="https://app.example.com/evaluations/",
            error_message=None,
        )
        # The error block is only rendered when error_message is truthy
        assert "Error:" not in data.html_content
