from typing import Any
from unittest.mock import MagicMock, patch

from app.core.telemetry.sentry import filters as sentry_filters


class TestBeforeSendErrorFilter:
    def test_drops_probe_scanner_event(self) -> None:
        event = {"transaction": "GET", "request": {"url": "http://x/health"}}
        assert sentry_filters.before_send_error_filter(event, {}) is None

    def test_scrubs_pii_when_send_default_pii_off(self) -> None:
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
        assert headers["Authorization"] == sentry_filters._REDACTED
        assert headers["Cookie"] == sentry_filters._REDACTED
        assert headers["Set-Cookie"] == sentry_filters._REDACTED
        assert headers["X-API-KEY"] == sentry_filters._REDACTED
        assert headers["User-Agent"] == "curl/8"
        assert result["request"]["query_string"] == sentry_filters._REDACTED
        assert result["request"]["cookies"] == sentry_filters._REDACTED
        assert result["request"]["data"] == sentry_filters._REDACTED

    def test_passes_normal_event_through(self) -> None:
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

    def test_keeps_headers_when_send_default_pii_on(self) -> None:
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

    def test_returns_event_on_internal_exception(self) -> None:
        bad_event = MagicMock()
        bad_event.get.side_effect = RuntimeError("malformed")

        assert sentry_filters.before_send_error_filter(bad_event, {}) is bad_event

    def test_scrubs_request_body_when_send_default_pii_on(self) -> None:
        event = {
            "transaction": "POST /api/v1/llm/generate",
            "request": {
                "url": "http://x/api/v1/llm/generate",
                "data": {"query": {"input": "what is my aadhaar number"}},
            },
        }
        with patch.object(sentry_filters.settings, "SENTRY_SEND_DEFAULT_PII", True):
            result = sentry_filters.before_send_error_filter(event, {})

        assert result["request"]["data"] == sentry_filters._REDACTED

    def test_scrubs_genai_content_from_error_event(self) -> None:
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
    def test_drops_content_keys_and_unpacked_subkeys(self) -> None:
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

    def test_is_case_insensitive(self) -> None:
        payload = {"GEN_AI.Response.Text": "hello"}
        sentry_filters.scrub_genai_content(payload)

        assert payload == {}

    def test_walks_nested_lists_and_dicts(self) -> None:
        payload = {
            "spans": [
                {"data": {"gen_ai.response.text": "hello", "gen_ai.system": "openai"}}
            ]
        }
        sentry_filters.scrub_genai_content(payload)

        assert payload["spans"][0]["data"] == {"gen_ai.system": "openai"}

    def test_stops_at_max_depth(self) -> None:
        deepest: dict = {"gen_ai.response.text": "hello"}
        payload: dict = deepest
        for _ in range(sentry_filters._GENAI_SCRUB_MAX_DEPTH + 2):
            payload = {"nested": payload}

        sentry_filters.scrub_genai_content(payload)

        assert deepest == {"gen_ai.response.text": "hello"}


class TestBeforeSendTransactionFilter:
    def test_scrubs_genai_content_from_spans(self) -> None:
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

    def test_keeps_gen_ai_span_itself(self) -> None:
        event = {
            "transaction": "POST /api/v1/llm/generate",
            "spans": [{"op": "gen_ai.chat", "description": "chat gpt-4o", "data": {}}],
        }
        result = sentry_filters.before_send_transaction_filter(event, {})

        assert result is not None
        assert len(result["spans"]) == 1


class TestBeforeSendLogFilter:
    def test_scrubs_genai_attributes(self) -> None:
        log = {
            "body": "[execute] done",
            "attributes": {
                "gen_ai.response.text": "model answer",
                "org_id": "1",
            },
        }
        assert sentry_filters.before_send_log_filter(log, {}) is log
        assert log["attributes"] == {"org_id": "1"}

    def test_returns_log_on_internal_exception(self) -> None:
        bad_log = MagicMock()
        bad_log.get.side_effect = RuntimeError("malformed")

        assert sentry_filters.before_send_log_filter(bad_log, {}) is bad_log


class TestGenaiPrivacyIntegrations:
    def test_all_returned_integrations_disable_prompts(self) -> None:
        integrations = sentry_filters.genai_privacy_integrations()

        assert integrations
        assert all(i.include_prompts is False for i in integrations)

    def test_skips_integrations_whose_sdk_is_missing(self) -> None:
        with patch.object(
            sentry_filters,
            "_GENAI_INTEGRATIONS",
            (("app.core.does_not_exist", "MissingIntegration"),),
        ):
            assert sentry_filters.genai_privacy_integrations() == []


class TestLlmJobKwargsRedaction:
    @staticmethod
    def _event_with_celery_job(
        task_name: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "extra": {
                "celery-job": {
                    "task_name": task_name,
                    "args": [],
                    "kwargs": kwargs,
                }
            }
        }

    def test_redacts_query_for_llm_job(self) -> None:
        event = self._event_with_celery_job(
            "app.celery.tasks.job_execution.run_llm_job",
            {
                "job_id": "b9ce6621-9b5a-45c4-969f-df7613ff7dc4",
                "organization_id": 1,
                "project_id": 1,
                "request_data": {
                    "callback_url": "https://webhooksite.net/some-id",
                    "config": {"blob": {"completion": {"provider": "openai"}}},
                    "query": {
                        "input": {
                            "type": "text",
                            "content": {
                                "value": "Amit Gupta phone number is 919611188278"
                            },
                        }
                    },
                    "request_metadata": None,
                },
            },
        )

        result = sentry_filters.before_send_error_filter(event, {})

        request_data = result["extra"]["celery-job"]["kwargs"]["request_data"]
        assert request_data["query"] == sentry_filters._REDACTED
        assert request_data["callback_url"] == sentry_filters._REDACTED
        assert request_data["config"] == {
            "blob": {"completion": {"provider": "openai"}}
        }

    def test_redacts_query_for_chain_and_response_jobs(self) -> None:
        for task_name in (
            "app.celery.tasks.job_execution.run_llm_chain_job",
            "app.celery.tasks.job_execution.run_response_job",
        ):
            event = self._event_with_celery_job(
                task_name,
                {"request_data": {"query": {"input": "sensitive text"}}},
            )

            result = sentry_filters.before_send_error_filter(event, {})

            request_data = result["extra"]["celery-job"]["kwargs"]["request_data"]
            assert request_data["query"] == sentry_filters._REDACTED

    def test_ignores_unrelated_tasks(self) -> None:
        event = self._event_with_celery_job(
            "app.celery.tasks.job_execution.run_doctransform_job",
            {"request_data": {"query": {"input": "not an llm job"}}},
        )

        result = sentry_filters.before_send_error_filter(event, {})

        request_data = result["extra"]["celery-job"]["kwargs"]["request_data"]
        assert request_data["query"] == {"input": "not an llm job"}

    def test_passes_through_events_without_celery_job(self) -> None:
        event = {"message": "some unrelated error"}

        assert sentry_filters.before_send_error_filter(event, {}) is event
