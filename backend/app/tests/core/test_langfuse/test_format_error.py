"""Unit tests for format_langfuse_error.

Langfuse 4.x's ``ApiError.__str__`` dumps the full HTTP response including every
response header. ``format_langfuse_error`` strips that noise so user-facing and
log messages only carry the actionable ``status_code`` and ``body`` (matching
the 2.x string format).
"""

from langfuse.api import NotFoundError
from langfuse.api.core.api_error import ApiError

from app.core.langfuse.langfuse import format_langfuse_error

# The header dump is what bloats ApiError.__str__ in langfuse 4.x.
HEADERS = {
    "date": "Fri, 05 Jun 2026 11:34:41 GMT",
    "content-type": "application/json; charset=utf-8",
    "etag": '"8eha70alug1r"',
}
BODY = {"message": "Dataset not found", "error": "LangfuseNotFoundError"}


def test_strips_headers_from_api_error():
    """The formatted message keeps status_code and body but drops headers."""
    exc = ApiError(status_code=404, headers=HEADERS, body=BODY)

    formatted = format_langfuse_error(exc)

    assert (
        formatted == "status_code: 404, body: {'message': 'Dataset not found', "
        "'error': 'LangfuseNotFoundError'}"
    )
    assert "headers:" not in formatted
    assert "etag" not in formatted


def test_handles_not_found_error_subclass():
    """NotFoundError (an ApiError subclass, status 404) is also cleaned up."""
    exc = NotFoundError(body=BODY, headers=HEADERS)

    formatted = format_langfuse_error(exc)

    assert formatted.startswith("status_code: 404, body:")
    assert "headers:" not in formatted


def test_raw_str_would_include_headers():
    """Guards the regression: the default __str__ does leak headers."""
    exc = ApiError(status_code=404, headers=HEADERS, body=BODY)

    assert "headers:" in str(exc)
    assert "headers:" not in format_langfuse_error(exc)


def test_falls_back_to_str_for_non_api_exceptions():
    """Non-langfuse exceptions are returned via str() unchanged."""
    assert format_langfuse_error(ValueError("plain message")) == "plain message"
    assert format_langfuse_error(RuntimeError("boom")) == "boom"
