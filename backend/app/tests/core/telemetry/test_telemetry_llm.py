from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from opentelemetry.instrumentation.utils import is_http_instrumentation_enabled

from app.core.telemetry import llm as telemetry
from app.core.telemetry import metrics


def _active_sentry() -> MagicMock:
    fake = MagicMock()
    fake.get_client.return_value.is_active.return_value = True
    return fake


def _metric_calls(fake: MagicMock) -> dict[str, dict]:
    calls = fake.metrics.count.call_args_list + fake.metrics.distribution.call_args_list
    return {c.kwargs["name"]: c.kwargs for c in calls}


class TestRecordLlmCallMetrics:
    def test_started_emits_count_with_tenant_attrs(self) -> None:
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(metrics, "sentry_sdk", fake),
        ):
            telemetry.record_llm_call_started(
                "openai", "gpt-4o", "chat", organization_id=1, project_id=2
            )

        attrs = _metric_calls(fake)["llm.call.total"]["attributes"]
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["kaapi.organization_id"] == "1"
        assert attrs["kaapi.project_id"] == "2"

    def test_finished_emits_duration_tokens_and_error(self) -> None:
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(metrics, "sentry_sdk", fake),
        ):
            telemetry.record_llm_call_finished(
                "openai",
                "gpt-4o",
                "chat",
                duration_ms=120.5,
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                error=True,
            )

        names = _metric_calls(fake)
        assert names["llm.call.duration"]["value"] == 120.5
        assert names["llm.call.errors"]["value"] == 1
        assert names["llm.tokens.input"]["value"] == 10
        assert names["llm.tokens.output"]["value"] == 20
        assert names["llm.tokens.total"]["value"] == 30
        assert "kaapi.organization_id" not in names["llm.call.duration"]["attributes"]

    def test_noop_when_otel_disabled(self) -> None:
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch.object(metrics, "sentry_sdk", fake),
        ):
            telemetry.record_llm_call_started("openai", "gpt-4o", "chat")
            telemetry.record_llm_call_finished("openai", "gpt-4o", "chat", 1.0)

        fake.metrics.count.assert_not_called()
        fake.metrics.distribution.assert_not_called()


class TestGenAiSpanAttributes:
    def test_request_attributes_include_params_and_tools(self) -> None:
        span = MagicMock()
        telemetry.set_gen_ai_request_attributes(
            span,
            provider="openai",
            model="gpt-4o",
            operation="chat",
            organization_id=1,
            project_id=2,
            params={"temperature": 0.2, "max_tokens": None, "tools": [{"name": "t"}]},
        )
        set_calls = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert set_calls["gen_ai.request.model"] == "gpt-4o"
        assert set_calls["kaapi.organization_id"] == 1
        assert set_calls["gen_ai.request.temperature"] == 0.2
        assert "gen_ai.request.max_tokens" not in set_calls
        assert set_calls["gen_ai.request.available_tools"] == '[{"name": "t"}]'

    def test_request_attributes_skip_missing_model_and_tenant(self) -> None:
        span = MagicMock()
        telemetry.set_gen_ai_request_attributes(
            span,
            provider="openai",
            model="",
            operation="chat",
            organization_id=None,
            project_id=None,
        )
        keys = {c.args[0] for c in span.set_attribute.call_args_list}
        assert keys == {
            "gen_ai.system",
            "gen_ai.provider.name",
            "gen_ai.operation.name",
        }

    def test_response_attributes_with_usage_and_model(self) -> None:
        span = MagicMock()
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1, output_tokens=2, total_tokens=3, reasoning_tokens=4
            ),
            response=SimpleNamespace(model="gpt-4o-2024"),
        )
        telemetry.set_gen_ai_response_attributes(span, response=response)
        set_calls = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert set_calls["gen_ai.usage.total_tokens"] == 3
        assert set_calls["gen_ai.usage.output_tokens.reasoning"] == 4
        assert set_calls["gen_ai.response.model"] == "gpt-4o-2024"

    def test_response_attributes_without_usage(self) -> None:
        span = MagicMock()
        telemetry.set_gen_ai_response_attributes(
            span, response=SimpleNamespace(usage=None, response=None)
        )
        span.set_attribute.assert_not_called()


class TestSuppressHttpInstrumentation:
    def test_disables_http_instrumentation_inside_block_only(self) -> None:
        assert is_http_instrumentation_enabled()
        with telemetry.suppress_http_instrumentation():
            assert not is_http_instrumentation_enabled()
        assert is_http_instrumentation_enabled()
