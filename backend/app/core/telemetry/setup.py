import logging
import re

import sentry_sdk
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind, StatusCode, format_span_id
from sentry_sdk.integrations.opentelemetry import SentryPropagator, SentrySpanProcessor

from app.core.config import settings
from app.core.telemetry.context import LogContextFilter

logger = logging.getLogger(__name__)


def _should_drop_bare_http_trace(
    *, is_root: bool, kind: SpanKind, status_code: StatusCode, had_children: bool
) -> bool:
    """True for a root server span with no children and no error: nothing worth a trace."""
    return (
        is_root
        and not had_children
        and kind == SpanKind.SERVER
        and status_code != StatusCode.ERROR
    )


def _build_resource(service_name: str | None = None) -> Resource:
    return Resource.create(
        {
            SERVICE_NAME: service_name or settings.OTEL_SERVICE_NAME,
            "deployment.environment": settings.ENVIRONMENT,
            "service.version": settings.API_VERSION,
        }
    )


class _NoiseFilteringSpanProcessor(SentrySpanProcessor):
    """Drop childless, error-free HTTP root spans so bare single-span traces never ship."""

    def __init__(self) -> None:
        super().__init__()
        self._traces_with_children: set[int] = set()

    def on_start(self, otel_span, parent_context=None):  # type: ignore[override]
        parent = otel_span.parent
        if parent is not None and not parent.is_remote:
            self._traces_with_children.add(otel_span.get_span_context().trace_id)
        super().on_start(otel_span, parent_context)

    def on_end(self, otel_span) -> None:  # type: ignore[override]
        span_context = otel_span.get_span_context()
        parent = otel_span.parent
        is_root = parent is None or parent.is_remote
        had_children = span_context.trace_id in self._traces_with_children
        if is_root:
            self._traces_with_children.discard(span_context.trace_id)
        if _should_drop_bare_http_trace(
            is_root=is_root,
            kind=otel_span.kind,
            status_code=otel_span.status.status_code,
            had_children=had_children,
        ):
            self._forget_span(otel_span)
            return
        super().on_end(otel_span)

    def _forget_span(self, otel_span) -> None:  # type: ignore[no-untyped-def]
        """Same bookkeeping as SentrySpanProcessor.on_end, minus shipping the span."""
        span_id = format_span_id(otel_span.get_span_context().span_id)
        self.otel_span_map.pop(span_id, None)
        if otel_span.start_time is not None:
            started_minute = int(otel_span.start_time / 1e9 / 60)
            bucket = self.open_spans.get(started_minute)
            if bucket is not None:
                bucket.discard(span_id)
        self._prune_old_spans()


def setup_telemetry(service_name: str | None = None) -> None:
    """Initialize OTel tracing and bridge spans into Sentry; logs/metrics go via the SDK directly."""
    root_logger = logging.getLogger()
    log_context_filter = LogContextFilter()
    if not any(isinstance(f, LogContextFilter) for f in root_logger.filters):
        root_logger.addFilter(log_context_filter)
    for handler in root_logger.handlers:
        if not any(isinstance(f, LogContextFilter) for f in handler.filters):
            handler.addFilter(log_context_filter)

    if not settings.OTEL_ENABLED:
        logger.info("[setup_telemetry] OTEL_ENABLED is False, skipping")
        return

    resource = _build_resource(service_name)
    tracer_provider = TracerProvider(resource=resource)

    if settings.SENTRY_DSN:
        tracer_provider.add_span_processor(_NoiseFilteringSpanProcessor())
        # Downstream services extract sentry-trace, not W3C traceparent.
        set_global_textmap(SentryPropagator())

    trace.set_tracer_provider(tracer_provider)

    LoggingInstrumentor().instrument(set_logging_format=False)
    HTTPXClientInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    try:
        # Circular import fix
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        CeleryInstrumentor().instrument()
    except Exception:
        logger.exception("[setup_telemetry] Failed to instrument Celery")

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument()
    except Exception:
        logger.exception("[setup_telemetry] Failed to instrument Redis")

    try:
        from opentelemetry.instrumentation.botocore import BotocoreInstrumentor

        BotocoreInstrumentor().instrument()
    except Exception:
        logger.exception("[setup_telemetry] Failed to instrument botocore")

    logger.debug(
        "[setup_telemetry] OpenTelemetry initialized (service=%s, sink=Sentry)",
        service_name or settings.OTEL_SERVICE_NAME,
    )


def flush_telemetry(timeout_millis: int = 10000) -> None:
    """Force-flush OTel spans into Sentry, then flush Sentry's transport.

    Called from Celery task_postrun: workers can recycle via max_tasks_per_child
    before the SentrySpanProcessor's internal queue drains, otherwise dropping
    closing spans and ERROR breadcrumbs from the just-finished task.
    """
    if not settings.OTEL_ENABLED:
        return
    try:
        tp = trace.get_tracer_provider()
        if hasattr(tp, "force_flush"):
            tp.force_flush(timeout_millis=timeout_millis)
    except Exception:
        logger.exception("[flush_telemetry] Failed to flush tracer provider")

    try:
        if sentry_sdk.get_client().is_active():
            sentry_sdk.flush(timeout=timeout_millis / 1000)
    except Exception:
        logger.exception("[flush_telemetry] Failed to flush Sentry")


def instrument_app(app: object) -> None:
    """Instrument the FastAPI app. Call after the app is created."""
    if not settings.OTEL_ENABLED:
        return
    from app.core.middleware import SILENT_LOG_PATHS, TRACE_EXCLUDED_PATH_PREFIXES

    # Doc/schema paths are read off the app so they follow config.
    exact_paths = set(SILENT_LOG_PATHS)
    for attr in (
        "docs_url",
        "redoc_url",
        "openapi_url",
        "swagger_ui_oauth2_redirect_url",
    ):
        path = getattr(app, attr, None)
        if path:
            exact_paths.add(path)

    patterns = [rf"^{re.escape(p)}/?$" for p in exact_paths]
    patterns += [rf"^{re.escape(p)}" for p in TRACE_EXCLUDED_PATH_PREFIXES]
    excluded_urls = ",".join(sorted(patterns))
    FastAPIInstrumentor.instrument_app(  # type: ignore[arg-type]
        app,
        excluded_urls=excluded_urls,
    )
    logger.debug("[instrument_app] FastAPI instrumented with OpenTelemetry")
