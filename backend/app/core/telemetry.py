import json
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import sentry_sdk
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_HTTP_INSTRUMENTATION_KEY
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind, StatusCode, format_span_id

from app.core.config import settings

if TYPE_CHECKING:
    from app.models.llm.response import LLMCallResponse

logger = logging.getLogger(__name__)

# Postgres SQLSTATE codes surfaced as named Sentry tags; others pass through as the raw code.
NOTABLE_SQLSTATES: dict[str, str] = {
    "40P01": "deadlock_detected",
    "57014": "query_canceled",  # statement timeout
    "40001": "serialization_failure",
    "55P03": "lock_not_available",  # lock timeout
    "08006": "connection_failure",
    "08003": "connection_does_not_exist",
    "53300": "too_many_connections",
}

_DB_CONNECTION_EVENT_METRICS: dict[str, str] = {
    "opened": "db.connection.opened",
    "closed": "db.connection.closed",
    "invalidated": "db.connection.invalidated",
}

_DB_TRANSACTION_METRICS: dict[str, str] = {
    "commit": "db.transaction.commit",
    "rollback": "db.transaction.rollback",
}

_log_context_var: ContextVar[dict[str, str] | None] = ContextVar(
    "kaapi_log_context", default=None
)

# OTel instrumentation scope emitted by SQLAlchemyInstrumentor; used to filter its spans.
_SQLALCHEMY_SCOPE = "opentelemetry.instrumentation.sqlalchemy"

# When True in the current context, SQLAlchemy DB spans are dropped before reaching Sentry.
_suppress_db_spans_var: ContextVar[bool] = ContextVar(
    "kaapi_suppress_db_spans", default=False
)


def _should_drop_db_span(otel_span: object) -> bool:
    """True when DB-span suppression is active and `otel_span` is a SQLAlchemy span."""
    if not _suppress_db_spans_var.get():
        return False
    scope = getattr(otel_span, "instrumentation_scope", None)
    return scope is not None and getattr(scope, "name", None) == _SQLALCHEMY_SCOPE


def _should_drop_bare_http_trace(
    *, is_root: bool, kind: SpanKind, status_code: StatusCode, had_children: bool
) -> bool:
    """True for a root HTTP server span whose trace has no child spans and no error.

    These single-span transactions carry no work worth a trace; errors are kept so
    failures stay visible (5xx also surface in http.server.request.error metrics).
    """
    return (
        is_root
        and not had_children
        and kind == SpanKind.SERVER
        and status_code != StatusCode.ERROR
    )


def _emit_sentry_metric(
    metric_type: str,
    name: str,
    value: float,
    *,
    unit: str | None = None,
    attributes: dict[str, str | int | float] | None = None,
) -> None:
    """Best-effort Sentry metric emission. No-op if SDK is not active."""
    try:
        if not sentry_sdk.get_client().is_active():
            return
        if metric_type == "count":
            sentry_sdk.metrics.count(
                name=name, value=value, unit=unit, attributes=attributes
            )
        elif metric_type == "gauge":
            sentry_sdk.metrics.gauge(
                name=name, value=value, unit=unit, attributes=attributes
            )
        elif metric_type == "distribution":
            sentry_sdk.metrics.distribution(
                name=name, value=value, unit=unit, attributes=attributes
            )
    except Exception:
        logger.debug("[_emit_sentry_metric] Failed to emit %s (%s)", name, metric_type)


def set_request_log_context(
    org_id: int | None = None,
    project_id: int | None = None,
) -> None:
    """Attach org/project to the current request's log context and Sentry scope.

    Call once per authenticated request (from the auth dependency). All subsequent
    log records in this request will carry org_id and project_id automatically
    via LogContextFilter — no need to add them to individual log statements.
    """
    current = _log_context_var.get() or {}
    payload = dict(current)
    if org_id is not None:
        payload["org_id"] = str(org_id)
    if project_id is not None:
        payload["project_id"] = str(project_id)
    _log_context_var.set(payload)

    try:
        if sentry_sdk.get_client().is_active():
            if org_id is not None:
                sentry_sdk.set_tag("tenant.org_id", str(org_id))
            if project_id is not None:
                sentry_sdk.set_tag("tenant.project_id", str(project_id))
    except Exception:
        pass


@contextmanager
def log_context(
    *, tag: str | None = None, **fields: str | int | float | bool | None
) -> Iterator[None]:
    """Attach structured log context for the current execution scope.

    Example:
        with log_context(tag="llm-call", job_id=job_id):
            logger.info("...")
    """
    current = _log_context_var.get() or {}
    payload = dict(current)
    if tag:
        payload["tag"] = tag
        payload.setdefault("system", tag)
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = str(value)
    token = _log_context_var.set(payload)
    try:
        yield
    finally:
        _log_context_var.reset(token)


class LogContextFilter(logging.Filter):
    """Attach structured context fields from `log_context(...)` to LogRecords."""

    _LLM_CALL_PREFIXES = (
        "app.services.llm",
        "app.api.routes.llm",
        "app.crud.llm",
    )
    _COLLECTION_PREFIXES = (
        "app.services.collections",
        "app.api.routes.collections",
        "app.crud.collection",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        context_payload = _log_context_var.get()
        has_explicit_tag = False
        if context_payload:
            for key, value in context_payload.items():
                setattr(record, key, value)
            has_explicit_tag = bool(context_payload.get("tag"))

        if not has_explicit_tag:
            logger_name = record.name or ""
            if logger_name.startswith(self._LLM_CALL_PREFIXES):
                record.tag = "llm-call"
                if not hasattr(record, "system"):
                    record.system = "llm-call"
            elif logger_name.startswith(self._COLLECTION_PREFIXES):
                record.tag = "collection"
                if not hasattr(record, "system"):
                    record.system = "collection"

        if not hasattr(record, "lifecycle"):
            span = trace.get_current_span()
            if span is not None and span.is_recording():
                span_name = getattr(span, "name", None)
                if isinstance(span_name, str) and span_name:
                    record.lifecycle = span_name
        return True


def _build_resource(service_name: str | None = None) -> Resource:
    return Resource.create(
        {
            SERVICE_NAME: service_name or settings.OTEL_SERVICE_NAME,
            "deployment.environment": settings.ENVIRONMENT,
            "service.version": settings.API_VERSION,
        }
    )


def setup_telemetry(service_name: str | None = None) -> None:
    """Initialize OpenTelemetry tracing and bridge spans into Sentry.

    Sentry is the single sink:
    - Traces:  OTel TracerProvider -> SentrySpanProcessor -> Sentry
    - Logs:    stdlib logging -> Sentry LoggingIntegration (configured at
               sentry_sdk.init time; see app/main.py and celery_app.py)
    - Metrics: code calls _emit_sentry_metric / sentry_sdk.metrics.* directly

    Args:
        service_name: Override OTEL_SERVICE_NAME (e.g. "kaapi-celery" in workers).
    """
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

    # Bridge OTel spans into Sentry as Sentry transactions and spans, with full attribute and error capture.
    if settings.SENTRY_DSN:
        from sentry_sdk.integrations.opentelemetry import SentrySpanProcessor

        class _NoiseFilteringSpanProcessor(SentrySpanProcessor):
            """Keep Sentry traces meaningful: drop suppressed DB spans and empty HTTP traces.

            - DB: while suppress_db_instrumentation() is active, SQLAlchemy spans are
              skipped by scope (HTTP/Requests keep flowing).
            - HTTP: a root server span whose trace gathered no child spans (and no error)
              is dropped at on_end — it never reaches Sentry as a bare single-span trace.
            """

            def __init__(self) -> None:
                super().__init__()
                self._traces_with_children: set[int] = set()

            def on_start(self, otel_span, parent_context=None):  # type: ignore[override]
                if _should_drop_db_span(otel_span):
                    return
                parent = otel_span.parent
                if parent is not None and not parent.is_remote:
                    self._traces_with_children.add(
                        otel_span.get_span_context().trace_id
                    )
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
                    self.otel_span_map.pop(format_span_id(span_context.span_id), None)
                    return
                super().on_end(otel_span)

        tracer_provider.add_span_processor(_NoiseFilteringSpanProcessor())

    trace.set_tracer_provider(tracer_provider)

    # Auto-instrumentation — generates OTel spans the SentrySpanProcessor forwards
    LoggingInstrumentor().instrument(set_logging_format=False)
    HTTPXClientInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    try:
        # Circular import fix
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        CeleryInstrumentor().instrument()
    except Exception:
        logger.exception("[setup_telemetry] Failed to instrument Celery")

    logger.debug(
        "[setup_telemetry] OpenTelemetry initialized (service=%s, sink=Sentry)",
        service_name or settings.OTEL_SERVICE_NAME,
    )


def _llm_call_attrs(
    provider: str,
    model: str,
    operation: str,
    organization_id: int | None,
    project_id: int | None,
) -> dict[str, str]:
    attrs: dict[str, str] = {
        "gen_ai.system": provider,
        "gen_ai.request.model": model,
        "gen_ai.operation.name": operation,
    }
    if organization_id is not None:
        attrs["kaapi.organization_id"] = str(organization_id)
    if project_id is not None:
        attrs["kaapi.project_id"] = str(project_id)
    return attrs


def record_llm_call_started(
    provider: str,
    model: str,
    operation: str,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> None:
    """Emit LLM call-start metric to Sentry."""
    if not settings.OTEL_ENABLED:
        return
    attrs = _llm_call_attrs(provider, model, operation, organization_id, project_id)
    _emit_sentry_metric("count", "llm.call.total", 1, attributes=attrs)


def record_llm_call_finished(
    provider: str,
    model: str,
    operation: str,
    duration_ms: float,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    error: bool = False,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> None:
    """Emit LLM call-completion metrics (latency, tokens, errors) to Sentry."""
    if not settings.OTEL_ENABLED:
        return
    attrs = _llm_call_attrs(provider, model, operation, organization_id, project_id)

    _emit_sentry_metric(
        "distribution",
        "llm.call.duration",
        duration_ms,
        unit="millisecond",
        attributes=attrs,
    )
    if error:
        _emit_sentry_metric("count", "llm.call.errors", 1, attributes=attrs)
    if input_tokens is not None:
        _emit_sentry_metric("count", "llm.tokens.input", input_tokens, attributes=attrs)
    if output_tokens is not None:
        _emit_sentry_metric(
            "count", "llm.tokens.output", output_tokens, attributes=attrs
        )
    if total_tokens is not None:
        _emit_sentry_metric("count", "llm.tokens.total", total_tokens, attributes=attrs)


def set_gen_ai_request_attributes(
    span: trace.Span,
    *,
    provider: str,
    model: str,
    operation: str,
    organization_id: int | None,
    project_id: int | None,
    params: dict[str, Any] | None = None,
) -> None:
    """Set OTel GenAI request attributes on `span` (semantic-convention keys + kaapi ids)."""
    span.set_attribute("gen_ai.system", provider)
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.operation.name", operation)
    if model:
        span.set_attribute("gen_ai.request.model", model)
    if organization_id is not None:
        span.set_attribute("kaapi.organization_id", organization_id)
        span.set_attribute("gen_ai.request.organization_id", organization_id)
    if project_id is not None:
        span.set_attribute("kaapi.project_id", project_id)
        span.set_attribute("gen_ai.request.project_id", project_id)

    params = params or {}
    for attr_key, param_key in (
        ("gen_ai.request.temperature", "temperature"),
        ("gen_ai.request.max_tokens", "max_tokens"),
        ("gen_ai.request.top_p", "top_p"),
        ("gen_ai.request.presence_penalty", "presence_penalty"),
        ("gen_ai.request.frequency_penalty", "frequency_penalty"),
    ):
        if param_key in params:
            span.set_attribute(attr_key, params.get(param_key))

    tools = params.get("tools")
    if tools is not None:
        span.set_attribute("gen_ai.request.available_tools", json.dumps(tools))


def set_gen_ai_response_attributes(
    span: trace.Span, *, response: "LLMCallResponse"
) -> None:
    """Set OTel GenAI response attributes (usage, model) on `span`."""
    usage = response.usage
    if usage:
        span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        span.set_attribute("gen_ai.usage.total_tokens", usage.total_tokens)
        if getattr(usage, "reasoning_tokens", None) is not None:
            span.set_attribute(
                "gen_ai.usage.output_tokens.reasoning", usage.reasoning_tokens
            )

    if response.response and response.response.model:
        span.set_attribute("gen_ai.response.model", response.response.model)


@contextmanager
def suppress_http_instrumentation() -> Iterator[None]:
    """Suppress OTel HTTP client auto-instrumentation for the wrapped block.

    Used around LLM provider calls so the outbound HTTP call does not emit a
    redundant child span — the LLM-level span already carries the trace.
    """
    token = otel_context.attach(
        otel_context.set_value(_SUPPRESS_HTTP_INSTRUMENTATION_KEY, True)
    )
    try:
        yield
    finally:
        otel_context.detach(token)


@contextmanager
def suppress_db_instrumentation() -> Iterator[None]:
    """Drop SQLAlchemy DB spans from the Sentry trace for the wrapped block.

    Wrap LLM job execution so its DB reads/writes do not clutter the LLM waterfall.
    Only DB spans are filtered (by _NoiseFilteringSpanProcessor via instrumentation
    scope) — HTTP/Requests instrumentation stays active. Trade-off: the wrapped
    DB queries also drop from the Sentry Queries page.
    """
    token = _suppress_db_spans_var.set(True)
    try:
        yield
    finally:
        _suppress_db_spans_var.reset(token)


def record_db_query_failed(
    *,
    operation: str | None = None,
    sqlstate: str | None = None,
) -> None:
    """Emit a DB query-failure counter to Sentry. Per-query duration/throughput come from spans."""
    if not settings.OTEL_ENABLED:
        return

    attrs: dict[str, str | int | float] = {}
    if operation:
        attrs["db.operation"] = operation
    if sqlstate:
        attrs["db.sqlstate"] = sqlstate

    _emit_sentry_metric("count", "db.query.failed", 1, attributes=attrs)


def record_db_slow_query(operation: str | None = None) -> None:
    """Emit a slow-query counter to Sentry (queries at/above settings.DB_SLOW_QUERY_MS)."""
    if not settings.OTEL_ENABLED:
        return

    attrs: dict[str, str | int | float] = {}
    if operation:
        attrs["db.operation"] = operation

    _emit_sentry_metric("count", "db.query.slow", 1, attributes=attrs)


def record_db_connection_event(event: str) -> None:
    """Emit a connection-lifecycle counter (opened/closed/invalidated) to Sentry."""
    if not settings.OTEL_ENABLED:
        return

    metric = _DB_CONNECTION_EVENT_METRICS.get(event)
    if metric is None:
        return
    _emit_sentry_metric("count", metric, 1)


def record_db_transaction(outcome: str) -> None:
    """Emit a transaction-outcome counter (commit/rollback) to Sentry for ratio tracking."""
    if not settings.OTEL_ENABLED:
        return

    metric = _DB_TRANSACTION_METRICS.get(outcome)
    if metric is None:
        return
    _emit_sentry_metric("count", metric, 1)


def _tag_db_error(sqlstate: str | None) -> None:
    """Tag the active Sentry scope + span with a DB error's Postgres SQLSTATE."""
    if not sqlstate:
        return
    try:
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("db.sqlstate", sqlstate)
        if sentry_sdk.get_client().is_active():
            sentry_sdk.set_tag("db.system", "postgresql")
            sentry_sdk.set_tag("db.sqlstate", sqlstate)
            name = NOTABLE_SQLSTATES.get(sqlstate)
            if name:
                sentry_sdk.set_tag("db.error.name", name)
    except Exception:
        logger.debug("[_tag_db_error] Failed to tag DB error | sqlstate: %s", sqlstate)


def record_db_pool_stats(
    *,
    active: int,
    idle: int,
    total: int,
    overflow: int,
) -> None:
    """Emit SQLAlchemy pool stats as Sentry gauges."""
    if not settings.OTEL_ENABLED:
        return

    _emit_sentry_metric("gauge", "db.pool.active", active)
    _emit_sentry_metric("gauge", "db.pool.idle", idle)
    _emit_sentry_metric("gauge", "db.pool.total", total)
    _emit_sentry_metric("gauge", "db.pool.overflow", overflow)


def record_stale_pending_jobs(
    *,
    table: str,
    status: str,
    stale_count: int,
    oldest_age_seconds: int | None,
    job_type: str | None = None,
    action_type: str | None = None,
    dimensional: bool = False,
) -> None:
    """Emit aggregate pending-job monitor metrics to Sentry.

    When ``dimensional`` is True, the metric is emitted under a separate
    name so per-group counts (grouped by job_type/action_type) do not get
    summed together with the table-level count in Sentry dashboards.
    """
    attrs: dict[str, str] = {
        "job.table": table,
        "job.status": status,
    }
    if job_type:
        attrs["job.type"] = job_type
    if action_type:
        attrs["job.action_type"] = action_type

    count_metric = (
        "jobs.pending.stale.by_dimension.count"
        if dimensional
        else "jobs.pending.stale.count"
    )
    age_metric = (
        "jobs.pending.oldest_age_seconds.by_dimension"
        if dimensional
        else "jobs.pending.oldest_age_seconds"
    )

    _emit_sentry_metric(
        "gauge",
        count_metric,
        stale_count,
        attributes=attrs,
    )
    if oldest_age_seconds is not None:
        _emit_sentry_metric(
            "gauge",
            age_metric,
            oldest_age_seconds,
            unit="second",
            attributes=attrs,
        )


def record_rate_threshold(
    *,
    project_id: int,
    project_name: str | None,
    category: str,
    request_count: int,
    threshold: int,
) -> None:
    """Emit rate threshold exceeded event to Sentry."""

    try:
        if not sentry_sdk.get_client().is_active():
            return
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("alert.type", "threshold_rate_monitor")
            scope.set_tag("tenant.project_id", project_id)
            scope.set_tag("route_category", category)
            scope.set_extra("request_count", request_count)
            scope.set_extra("threshold", threshold)
            sentry_sdk.capture_message(
                f"[Threshold-Monitor] {category} rate limit exceeded for project {project_id} | {project_name}: {request_count} req/min "
                f"(limit {threshold}/min)",
                level="warning",
            )
    except Exception as e:
        logger.exception("[record_rate_threshold] Failed to emit alert", exc_info=e)


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

    # Non-business request spans are noise: health/utility, FastAPI's own doc/schema
    # endpoints (read off the app so they track config), and cron polling (prefix).
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


def instrument_db_engine(engine: object) -> None:
    """Instrument a SQLAlchemy engine: query spans, slow-query/pool/connection/transaction metrics."""
    if not settings.OTEL_ENABLED:
        return
    if getattr(engine, "_kaapi_db_telemetry_instrumented", False):
        return

    # DB query spans -> SentrySpanProcessor -> Sentry Insights/Queries. ProxyTracer defers
    # to the real provider set later in setup_telemetry(), so load order here is safe.
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(engine=engine)
    except Exception:
        logger.exception(
            "[instrument_db_engine] Failed to load SQLAlchemy span instrumentation"
        )

    try:
        from sqlalchemy import ExceptionContext, event
    except Exception:
        logger.exception("[instrument_db_engine] Failed importing SQLAlchemy events")
        return

    def _pool_snapshot(pool: Any) -> tuple[int, int, int, int] | None:
        if not all(hasattr(pool, attr) for attr in ("checkedout", "size", "overflow")):
            return None
        active = int(pool.checkedout())
        idle = int(pool.checkedin()) if hasattr(pool, "checkedin") else 0
        configured_size = int(pool.size())
        overflow = int(pool.overflow())
        total = max(configured_size + overflow, active + idle)
        return active, idle, total, overflow

    def _emit_pool_metrics(pool: Any) -> None:
        snapshot = _pool_snapshot(pool)
        if not snapshot:
            return
        active, idle, total, overflow = snapshot
        record_db_pool_stats(active=active, idle=idle, total=total, overflow=overflow)

    @event.listens_for(engine, "before_cursor_execute")
    def _before_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ) -> None:
        del cursor, parameters, executemany
        # Timing only flags slow queries; per-query duration lives on the span.
        context._kaapi_db_started_at = time.perf_counter()
        context._kaapi_db_operation = (
            str(statement).split(None, 1)[0].upper() if statement else "UNKNOWN"
        )
        _emit_pool_metrics(conn.engine.pool)

    @event.listens_for(engine, "after_cursor_execute")
    def _after_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ) -> None:
        del cursor, statement, parameters, executemany
        started_at = getattr(context, "_kaapi_db_started_at", None)
        if started_at is not None:
            duration_ms = (time.perf_counter() - started_at) * 1000
            if duration_ms >= settings.DB_SLOW_QUERY_MS:
                record_db_slow_query(getattr(context, "_kaapi_db_operation", None))
        _emit_pool_metrics(conn.engine.pool)

    @event.listens_for(engine, "handle_error")
    def _handle_error(exception_context: ExceptionContext) -> None:
        context = exception_context.execution_context
        operation = (
            getattr(context, "_kaapi_db_operation", None)
            if context is not None
            else None
        )
        sqlstate = getattr(exception_context.original_exception, "sqlstate", None)
        _tag_db_error(sqlstate)
        record_db_query_failed(operation=operation, sqlstate=sqlstate)

    @event.listens_for(engine.pool, "checkout")
    def _on_checkout(dbapi_connection, connection_record, connection_proxy) -> None:
        del dbapi_connection, connection_record, connection_proxy
        _emit_pool_metrics(engine.pool)

    @event.listens_for(engine.pool, "checkin")
    def _on_checkin(dbapi_connection, connection_record) -> None:
        del dbapi_connection, connection_record
        _emit_pool_metrics(engine.pool)

    @event.listens_for(engine.pool, "connect")
    def _on_connect(dbapi_connection, connection_record) -> None:
        del dbapi_connection, connection_record
        record_db_connection_event("opened")

    @event.listens_for(engine.pool, "close")
    def _on_close(dbapi_connection, connection_record) -> None:
        del dbapi_connection, connection_record
        record_db_connection_event("closed")

    @event.listens_for(engine.pool, "invalidate")
    def _on_invalidate(dbapi_connection, connection_record, exception) -> None:
        del dbapi_connection, connection_record, exception
        record_db_connection_event("invalidated")

    @event.listens_for(engine, "commit")
    def _on_commit(conn) -> None:
        del conn
        record_db_transaction("commit")

    @event.listens_for(engine, "rollback")
    def _on_rollback(conn) -> None:
        del conn
        record_db_transaction("rollback")

    engine._kaapi_db_telemetry_instrumented = True
    logger.debug("[instrument_db_engine] SQLAlchemy DB telemetry enabled")
