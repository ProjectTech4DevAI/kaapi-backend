"""Tests for the Sentry before_send error filter in core/sentry_filters.py.

settings is patched via patch.object; no real Sentry connection is used. The
filter is a pure function over the event dict, so cases assert on the returned
event (or None) and on in-place PII scrubbing.
"""

from unittest.mock import MagicMock, patch

from app.core import sentry_filters


class TestBeforeSendErrorFilter:
    def test_drops_probe_scanner_event(self):
        event = {"transaction": "GET", "request": {"url": "http://x/health"}}
        assert sentry_filters.before_send_error_filter(event, {}) is None

    def test_scrubs_pii_when_send_default_pii_off(self):
        event = {
            "transaction": "POST /api/v1/llm/generate",
            "request": {
                "url": "http://x/api/v1/llm/generate",
                "headers": {
                    "Authorization": "Bearer secret",
                    "Cookie": "session=abc",
                    "Set-Cookie": "session=abc",
                    "X-API-KEY": "sk-123",
                    "User-Agent": "curl/8",
                },
                "query_string": "token=secret",
                "cookies": {"session": "abc"},
                "data": {"password": "hunter2"},
            },
        }
        with patch.object(sentry_filters.settings, "SENTRY_SEND_DEFAULT_PII", False):
            result = sentry_filters.before_send_error_filter(event, {})

        assert result is event
        headers = result["request"]["headers"]
        assert headers["Authorization"] == sentry_filters._SCRUBBED
        assert headers["Cookie"] == sentry_filters._SCRUBBED
        assert headers["Set-Cookie"] == sentry_filters._SCRUBBED
        assert headers["X-API-KEY"] == sentry_filters._SCRUBBED
        assert headers["User-Agent"] == "curl/8"
        assert result["request"]["query_string"] == sentry_filters._SCRUBBED
        assert result["request"]["cookies"] == sentry_filters._SCRUBBED
        assert result["request"]["data"] == sentry_filters._SCRUBBED

    def test_passes_normal_event_through(self):
        event = {
            "transaction": "POST /api/v1/llm/generate",
            "request": {
                "url": "http://x/api/v1/llm/generate",
                "headers": {"User-Agent": "curl/8"},
            },
        }
        with patch.object(sentry_filters.settings, "SENTRY_SEND_DEFAULT_PII", False):
            result = sentry_filters.before_send_error_filter(event, {})

        assert result is event
        assert result["request"]["headers"]["User-Agent"] == "curl/8"

    def test_keeps_headers_when_send_default_pii_on(self):
        event = {
            "transaction": "POST /api/v1/llm/generate",
            "request": {
                "url": "http://x/api/v1/llm/generate",
                "headers": {"Authorization": "Bearer secret"},
                "cookies": {"session": "abc"},
            },
        }
        with patch.object(sentry_filters.settings, "SENTRY_SEND_DEFAULT_PII", True):
            result = sentry_filters.before_send_error_filter(event, {})

        assert result["request"]["headers"]["Authorization"] == "Bearer secret"
        assert result["request"]["cookies"] == {"session": "abc"}

    def test_returns_event_on_internal_exception(self):
        bad_event = MagicMock()
        bad_event.get.side_effect = RuntimeError("malformed")

        assert sentry_filters.before_send_error_filter(bad_event, {}) is bad_event
