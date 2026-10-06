import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from opentelemetry.trace import SpanKind, StatusCode, format_span_id
from opentelemetry.util.http import ExcludeList
from sentry_sdk.integrations.opentelemetry import SentrySpanProcessor

from app.core.telemetry import setup as telemetry


class TestSetupTelemetryInstrumentors:
    def test_enabled_calls_instrument_on_both(self) -> None:
        redis, botocore = MagicMock(), MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry.settings, "SENTRY_DSN", None),
            patch.object(telemetry, "TracerProvider", MagicMock()),
            patch.object(telemetry.trace, "set_tracer_provider", MagicMock()),
            patch.object(telemetry, "LoggingInstrumentor", MagicMock()),
            patch.object(telemetry, "HTTPXClientInstrumentor", MagicMock()),
            patch.object(telemetry, "RequestsInstrumentor", MagicMock()),
            patch(
                "opentelemetry.instrumentation.celery.CeleryInstrumentor", MagicMock()
            ),
            patch("opentelemetry.instrumentation.redis.RedisInstrumentor", redis),
            patch(
                "opentelemetry.instrumentation.botocore.BotocoreInstrumentor", botocore
            ),
        ):
            telemetry.setup_telemetry()

        redis.return_value.instrument.assert_called_once()
        botocore.return_value.instrument.assert_called_once()

    def test_redis_failure_does_not_propagate_and_botocore_still_instruments(
        self,
    ) -> None:
        redis = MagicMock()
        redis.return_value.instrument.side_effect = RuntimeError("boom")
        botocore = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry.settings, "SENTRY_DSN", None),
            patch.object(telemetry, "TracerProvider", MagicMock()),
            patch.object(telemetry.trace, "set_tracer_provider", MagicMock()),
            patch.object(telemetry, "LoggingInstrumentor", MagicMock()),
            patch.object(telemetry, "HTTPXClientInstrumentor", MagicMock()),
            patch.object(telemetry, "RequestsInstrumentor", MagicMock()),
            patch(
                "opentelemetry.instrumentation.celery.CeleryInstrumentor", MagicMock()
            ),
            patch("opentelemetry.instrumentation.redis.RedisInstrumentor", redis),
            patch(
                "opentelemetry.instrumentation.botocore.BotocoreInstrumentor", botocore
            ),
        ):
            telemetry.setup_telemetry()

        botocore.return_value.instrument.assert_called_once()

    def test_noop_when_otel_disabled(self) -> None:
        redis, botocore = MagicMock(), MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch("opentelemetry.instrumentation.redis.RedisInstrumentor", redis),
            patch(
                "opentelemetry.instrumentation.botocore.BotocoreInstrumentor", botocore
            ),
        ):
            telemetry.setup_telemetry()

        redis.assert_not_called()
        botocore.assert_not_called()


class TestShouldDropBareHttpTrace:
    def test_drops_root_http_span_with_no_children(self) -> None:
        assert telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.SERVER,
            status_code=StatusCode.UNSET,
            had_children=False,
        )

    def test_keeps_when_trace_has_children(self) -> None:
        assert not telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.SERVER,
            status_code=StatusCode.UNSET,
            had_children=True,
        )

    def test_keeps_error_trace_even_without_children(self) -> None:
        assert not telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.SERVER,
            status_code=StatusCode.ERROR,
            had_children=False,
        )

    def test_keeps_non_root_span(self) -> None:
        assert not telemetry._should_drop_bare_http_trace(
            is_root=False,
            kind=SpanKind.SERVER,
            status_code=StatusCode.UNSET,
            had_children=False,
        )

    def test_keeps_non_server_root(self) -> None:
        assert not telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.INTERNAL,
            status_code=StatusCode.UNSET,
            had_children=False,
        )


class TestNoiseFilteringSpanProcessor:
    @staticmethod
    def _root_server_span(span_id: int, start_time_ns: int) -> MagicMock:
        span = MagicMock()
        span.parent = None
        span.kind = SpanKind.SERVER
        span.status.status_code = StatusCode.UNSET
        span.start_time = start_time_ns
        span.get_span_context.return_value = SimpleNamespace(
            trace_id=0xABC, span_id=span_id, is_valid=True
        )
        return span

    def test_dropped_root_span_is_removed_from_open_spans(self) -> None:
        processor = telemetry._NoiseFilteringSpanProcessor()
        start_time_ns = int(time.time() * 1e9)
        span = self._root_server_span(span_id=0x1, start_time_ns=start_time_ns)
        span_id = format_span_id(0x1)
        started_minute = int(start_time_ns / 1e9 / 60)
        processor.otel_span_map[span_id] = MagicMock()
        processor.open_spans[started_minute] = {span_id}

        with patch.object(SentrySpanProcessor, "on_end") as base_on_end:
            processor.on_end(span)

        base_on_end.assert_not_called()
        assert span_id not in processor.otel_span_map
        assert all(span_id not in bucket for bucket in processor.open_spans.values())


class TestInstrumentApp:
    def _app(self) -> FastAPI:
        return FastAPI(openapi_url="/api/v1/openapi.json")

    def _excluded(self) -> ExcludeList:
        instrument = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry.FastAPIInstrumentor, "instrument_app", instrument),
        ):
            telemetry.instrument_app(self._app())

        excluded_urls = instrument.call_args.kwargs["excluded_urls"]
        return ExcludeList(excluded_urls.split(","))

    def test_health_and_cron_paths_excluded_from_traces(self) -> None:
        el = self._excluded()
        assert el.url_disabled("/health")
        assert el.url_disabled("/api/v1/utils/health")
        assert el.url_disabled("/api/v1/cron/run_batches")

    def test_framework_doc_paths_excluded_from_traces(self) -> None:
        el = self._excluded()
        assert el.url_disabled("/docs")
        assert el.url_disabled("/redoc")
        assert el.url_disabled("/api/v1/openapi.json")
        assert el.url_disabled("/docs/oauth2-redirect")

    def test_real_endpoints_still_traced(self) -> None:
        el = self._excluded()
        assert not el.url_disabled("/api/v1/llm/generate")
        assert not el.url_disabled("/api/v1/cronies")
        assert not el.url_disabled("/api/v1/documents")

    def test_noop_when_otel_disabled(self) -> None:
        instrument = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch.object(telemetry.FastAPIInstrumentor, "instrument_app", instrument),
        ):
            telemetry.instrument_app(self._app())

        instrument.assert_not_called()
