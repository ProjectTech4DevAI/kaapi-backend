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

    def test_scrubs_request_body_when_send_default_pii_on(self):
        event = {
            "transaction": "POST /api/v1/llm/generate",
            "request": {
                "url": "http://x/api/v1/llm/generate",
                "data": {"query": {"input": "what is my aadhaar number"}},
            },
        }
        with patch.object(sentry_filters.settings, "SENTRY_SEND_DEFAULT_PII", True):
            result = sentry_filters.before_send_error_filter(event, {})

        assert result["request"]["data"] == sentry_filters._SCRUBBED

    def test_scrubs_genai_content_from_error_event(self):
        event = {
            "transaction": "POST /api/v1/llm/generate",
            "contexts": {
                "trace": {
                    "data": {
                        "gen_ai.request.messages": [{"content": "user secret"}],
                        "gen_ai.response.text": "model answer",
                        "gen_ai.request.model": "gpt-4o",
                    }
                }
            },
        }
        with patch.object(sentry_filters.settings, "SENTRY_SEND_DEFAULT_PII", False):
            result = sentry_filters.before_send_error_filter(event, {})

        data = result["contexts"]["trace"]["data"]
        assert "gen_ai.request.messages" not in data
        assert "gen_ai.response.text" not in data
        assert data["gen_ai.request.model"] == "gpt-4o"


class TestScrubGenaiContent:
    def test_drops_content_keys_and_unpacked_subkeys(self):
        payload = {
            "gen_ai.request.messages": ["hi"],
            "gen_ai.request.messages.0.content": "hi",
            "gen_ai.response.text": "hello",
            "ai.input_messages": ["hi"],
            "langfuse.trace.output": "hello",
            "gen_ai.usage.total_tokens": 42,
        }
        sentry_filters.scrub_genai_content(payload)

        assert payload == {"gen_ai.usage.total_tokens": 42}

    def test_is_case_insensitive(self):
        payload = {"GEN_AI.Response.Text": "hello"}
        sentry_filters.scrub_genai_content(payload)

        assert payload == {}

    def test_walks_nested_lists_and_dicts(self):
        payload = {
            "spans": [
                {"data": {"gen_ai.response.text": "hello", "gen_ai.system": "openai"}}
            ]
        }
        sentry_filters.scrub_genai_content(payload)

        assert payload["spans"][0]["data"] == {"gen_ai.system": "openai"}

    def test_stops_at_max_depth(self):
        deepest: dict = {"gen_ai.response.text": "hello"}
        payload: dict = deepest
        for _ in range(sentry_filters._GENAI_SCRUB_MAX_DEPTH + 2):
            payload = {"nested": payload}

        sentry_filters.scrub_genai_content(payload)

        assert deepest == {"gen_ai.response.text": "hello"}


class TestBeforeSendTransactionFilter:
    def test_scrubs_genai_content_from_spans(self):
        event = {
            "transaction": "POST /api/v1/llm/generate",
            "spans": [
                {
                    "op": "gen_ai.chat",
                    "description": "chat gpt-4o",
                    "data": {
                        "gen_ai.request.messages": [{"content": "user secret"}],
                        "gen_ai.response.text": "model answer",
                        "gen_ai.usage.total_tokens": 12,
                    },
                }
            ],
        }
        result = sentry_filters.before_send_transaction_filter(event, {})

        assert result is not None
        span_data = result["spans"][0]["data"]
        assert span_data == {"gen_ai.usage.total_tokens": 12}

    def test_keeps_gen_ai_span_itself(self):
        event = {
            "transaction": "POST /api/v1/llm/generate",
            "spans": [{"op": "gen_ai.chat", "description": "chat gpt-4o", "data": {}}],
        }
        result = sentry_filters.before_send_transaction_filter(event, {})

        assert result is not None
        assert len(result["spans"]) == 1


class TestBeforeSendLogFilter:
    def test_scrubs_genai_attributes(self):
        log = {
            "body": "[execute] done",
            "attributes": {
                "gen_ai.response.text": "model answer",
                "org_id": "1",
            },
        }
        assert sentry_filters.before_send_log_filter(log, {}) is log
        assert log["attributes"] == {"org_id": "1"}

    def test_returns_log_on_internal_exception(self):
        bad_log = MagicMock()
        bad_log.get.side_effect = RuntimeError("malformed")

        assert sentry_filters.before_send_log_filter(bad_log, {}) is bad_log


class TestGenaiPrivacyIntegrations:
    def test_all_returned_integrations_disable_prompts(self):
        integrations = sentry_filters.genai_privacy_integrations()

        assert integrations
        assert all(i.include_prompts is False for i in integrations)

    def test_skips_integrations_whose_sdk_is_missing(self):
        with patch.object(
            sentry_filters,
            "_GENAI_INTEGRATIONS",
            (("app.core.does_not_exist", "MissingIntegration"),),
        ):
            assert sentry_filters.genai_privacy_integrations() == []
