"""Provider error formatting. Payloads mirror real provider responses."""

import httpx
import httpx2
import pytest
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APIStatusError as AnthropicStatusError
from google.api_core import exceptions as gcs_exceptions
from google.genai import errors as genai_errors
from openai import BadRequestError as OpenAIBadRequestError

from app.services.assessment.api.provider_errors import (
    MAX_LISTED_REASONS,
    describe_exception,
    error_from_payload,
    error_from_vertex_status,
    format_provider_error,
    job_failure_reason,
    join_reasons,
    provider_label,
    redact_urls,
    row_error,
)


class TestFormatProviderError:
    def test_code_message_and_request_id(self) -> None:
        assert (
            format_provider_error(
                "anthropic", code="not_found_error", message="model: x", request_id="r1"
            )
            == "[ANTHROPIC] not_found_error: model: x (request_id=r1)"
        )

    def test_message_only(self) -> None:
        assert format_provider_error("openai", message="boom") == "[OPENAI] boom"

    def test_nothing_known(self) -> None:
        assert format_provider_error("openai") == "[OPENAI] unknown error"

    @pytest.mark.parametrize(
        ("provider", "label"),
        [
            ("openai", "OPENAI"),
            ("anthropic", "ANTHROPIC"),
            ("google", "GEMINI"),
            ("google-aistudio", "GEMINI"),
            ("google-gcp", "VERTEX"),
        ],
    )
    def test_labels(self, provider: str, label: str) -> None:
        assert provider_label(provider) == label


class TestErrorFromPayload:
    def test_anthropic_errored_result(self) -> None:
        payload = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Unable to download the file. Please verify the URL and try again.",
                "details": None,
            },
            "request_id": "req_011ChP9h",
        }
        assert error_from_payload("anthropic", payload) == (
            "[ANTHROPIC] invalid_request_error: Unable to download the file. Please "
            "verify the URL and try again. (request_id=req_011ChP9h)"
        )

    def test_gemini_row_error_maps_the_numeric_code(self) -> None:
        payload = {"code": 3, "message": "Request contains an invalid argument."}
        assert error_from_payload("google-aistudio", payload) == (
            "[GEMINI] INVALID_ARGUMENT: Request contains an invalid argument."
        )

    def test_openai_prefers_the_specific_code(self) -> None:
        payload = {
            "message": "Error while downloading file. Upstream status code: 403.",
            "type": "invalid_request_error",
            "param": "url",
            "code": "invalid_value",
        }
        assert error_from_payload("openai", payload) == (
            "[OPENAI] invalid_value: Error while downloading file. Upstream status "
            "code: 403. (param=url)"
        )

    def test_payload_without_message_is_kept_whole(self) -> None:
        assert error_from_payload("openai", {"code": 500}) == (
            '[OPENAI] 500: {"code": 500}'
        )

    def test_plain_string(self) -> None:
        assert error_from_payload("openai", "oops") == "[OPENAI] oops"

    def test_string_error_inside_an_envelope(self) -> None:
        assert error_from_payload("openai", {"error": "bad", "request_id": "r"}) == (
            "[OPENAI] bad (request_id=r)"
        )

    def test_formatted_string_is_not_relabelled(self) -> None:
        text = "[GEMINI] NOT_FOUND: gone"
        assert error_from_payload("google-aistudio", text) == text


class TestErrorFromVertexStatus:
    def test_bad_request_status_line(self) -> None:
        status = (
            'Bad Request: {"error": {"code": 400, "message": "Please use a valid role: '
            'user, model.", "status": "INVALID_ARGUMENT"}}'
        )
        assert error_from_vertex_status("google-gcp", status) == (
            "[VERTEX] INVALID_ARGUMENT: Please use a valid role: user, model."
        )

    def test_unparseable_status_kept_as_is(self) -> None:
        assert error_from_vertex_status("google-gcp", "quota") == "[VERTEX] quota"

    def test_non_string_status_does_not_raise(self) -> None:
        status = {"code": 3, "message": "bad part"}
        assert error_from_vertex_status("google-gcp", status) == (
            "[VERTEX] INVALID_ARGUMENT: bad part"
        )


class TestJoinReasons:
    def test_few_reasons_are_all_listed(self) -> None:
        assert join_reasons(["a", "b"]) == "a | b"

    def test_extra_reasons_are_counted_not_listed(self) -> None:
        reasons = [str(i) for i in range(MAX_LISTED_REASONS + 2)]
        assert join_reasons(reasons, separator="; ") == (
            "0; 1; 2; 3; 4; +2 more distinct error(s)"
        )


class TestRedactUrls:
    def test_signed_url_query_is_removed(self) -> None:
        text = "fetch failed: https://b.s3.amazonaws.com/a.png?X-Amz-Signature=abc&X-Amz-Credential=k"
        assert (
            redact_urls(text)
            == "fetch failed: https://b.s3.amazonaws.com/a.png?<redacted>"
        )

    def test_plain_url_is_untouched(self) -> None:
        assert (
            redact_urls("see https://example.com/a.png")
            == "see https://example.com/a.png"
        )


class TestDescribeException:
    def test_gemini_client_error(self) -> None:
        exc = genai_errors.ClientError(
            404,
            {
                "error": {
                    "code": 404,
                    "message": "models/gemini-does-not-exist is not found for API version v1beta",
                    "status": "NOT_FOUND",
                }
            },
        )
        assert describe_exception("google-aistudio", exc) == (
            "[GEMINI] NOT_FOUND: models/gemini-does-not-exist is not found for API "
            "version v1beta"
        )

    def test_anthropic_status_error(self) -> None:
        response = httpx2.Response(
            400,
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
            headers={"request-id": "req_a"},
        )
        exc = AnthropicStatusError(
            "Error code: 400",
            response=response,
            body={"error": {"type": "invalid_request_error", "message": "bad schema"}},
        )
        assert describe_exception("anthropic", exc) == (
            "[ANTHROPIC] invalid_request_error: bad schema (request_id=req_a)"
        )

    def test_anthropic_connection_error(self) -> None:
        exc = AnthropicConnectionError(
            request=httpx2.Request(
                "POST", "https://api.anthropic.com/v1/messages/batches"
            )
        )
        assert describe_exception("anthropic", exc) == (
            "[ANTHROPIC] APIConnectionError: Connection error."
        )

    def test_openai_status_error_with_bare_body(self) -> None:
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.openai.com/v1/batches"),
            headers={"x-request-id": "req_o"},
        )
        exc = OpenAIBadRequestError(
            "bad",
            response=response,
            body={
                "message": "Invalid model",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            },
        )
        assert describe_exception("openai", exc) == (
            "[OPENAI] model_not_found: Invalid model (param=model) (request_id=req_o)"
        )

    def test_gcs_error_wrapped_by_a_storage_error(self) -> None:
        cause = gcs_exceptions.Forbidden(
            "POST https://storage.googleapis.com/upload/storage/v1/b/x/o?uploadType=multipart",
            errors=[
                {
                    "message": "The billing account for the owning project is disabled in state closed",
                    "reason": "accountDisabled",
                }
            ],
        )
        wrapper = RuntimeError("GCS upload failed")
        wrapper.__cause__ = cause
        assert describe_exception("google-gcp", wrapper) == (
            "[VERTEX] accountDisabled: The billing account for the owning project is "
            "disabled in state closed"
        )

    def test_http_exception_uses_its_detail(self) -> None:
        from fastapi import HTTPException

        exc = HTTPException(
            404, "google-gcp credentials not configured for this project"
        )
        assert describe_exception("google-gcp", exc) == (
            "[VERTEX] 404: google-gcp credentials not configured for this project"
        )

    def test_unknown_exception(self) -> None:
        assert describe_exception("openai", RuntimeError("boom")) == (
            "[OPENAI] RuntimeError: boom"
        )


class TestRowError:
    def test_vertex_status_string_is_parsed(self) -> None:
        status = (
            'Bad Request: {"error": {"code": 400, "message": "Please use a valid '
            'role: user, model.", "status": "INVALID_ARGUMENT"}}'
        )
        assert row_error("google-gcp", status) == (
            "[VERTEX] INVALID_ARGUMENT: Please use a valid role: user, model."
        )

    def test_structured_error_is_formatted(self) -> None:
        assert row_error("google-aistudio", {"code": 5, "message": "gone"}) == (
            "[GEMINI] NOT_FOUND: gone"
        )


class TestJobFailureReason:
    def test_openai_batch_errors(self) -> None:
        status_result = {
            "error_message": "Batch failed",
            "provider_errors": [
                {
                    "code": "model_not_found",
                    "message": "The provided model 'gpt-x' is not supported by the Batch API.",
                    "param": "body.model",
                    "line": 1,
                }
            ],
        }
        assert job_failure_reason("openai", status_result) == (
            "[OPENAI] model_not_found: The provided model 'gpt-x' is not supported "
            "by the Batch API. (param=body.model, line=1)"
        )

    def test_gemini_job_error(self) -> None:
        status_result = {
            "error_message": "Batch JOB_STATE_FAILED",
            "provider_error": {"code": 7, "message": "permission denied on bucket"},
        }
        assert job_failure_reason("google-aistudio", status_result) == (
            "[GEMINI] PERMISSION_DENIED: permission denied on bucket"
        )

    def test_falls_back_to_the_adapter_message(self) -> None:
        assert job_failure_reason("openai", {"error_message": "Batch expired"}) == (
            "[OPENAI] Batch expired"
        )

    def test_nothing_to_report(self) -> None:
        assert job_failure_reason("openai", {}) is None
