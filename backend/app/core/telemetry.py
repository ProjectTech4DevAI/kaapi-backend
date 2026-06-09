import json
import logging
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

from app.core.config import settings

if TYPE_CHECKING:
    from app.models.llm.response import LLMCallResponse

logger = logging.getLogger(__name__)

_log_context_var: ContextVar[dict[str, str] | None] = ContextVar(
    "kaapi_log_context", default=None
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

        tracer_provider.add_span_processor(SentrySpanProcessor())

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


def record_db_query_finished(
    *,
    duration_ms: float,
    operation: str | None = None,
    error: bool = False,
) -> None:
    """Emit DB query metrics to Sentry."""
    if not settings.OTEL_ENABLED:
        return

    attrs: dict[str, str] = {}
    if operation:
        attrs["db.operation"] = operation

    _emit_sentry_metric("count", "db.query.total", 1, attributes=attrs)
    _emit_sentry_metric(
        "distribution",
        "db.query.duration",
        duration_ms,
        unit="millisecond",
        attributes=attrs,
    )
    if error:
        _emit_sentry_metric("count", "db.query.failed", 1, attributes=attrs)


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
    # Health checks are high-volume/noise and should not generate traces.
    FastAPIInstrumentor.instrument_app(  # type: ignore[arg-type]
        app,
        excluded_urls=r"^/health/?$",
    )
    logger.debug("[instrument_app] FastAPI instrumented with OpenTelemetry")


def instrument_db_engine(engine: object) -> None:
    """Instrument SQLAlchemy engine with DB query spans + pool gauges."""
    if not settings.OTEL_ENABLED:
        return
    if getattr(engine, "_kaapi_db_telemetry_instrumented", False):
        return

    try:
        from sqlalchemy import event
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
        duration_ms = (
            (time.perf_counter() - started_at) * 1000 if started_at is not None else 0.0
        )
        operation = getattr(context, "_kaapi_db_operation", None)
        record_db_query_finished(
            duration_ms=duration_ms, operation=operation, error=False
        )
        _emit_pool_metrics(conn.engine.pool)

    @event.listens_for(engine, "handle_error")
    def _handle_error(exception_context) -> None:
        context = exception_context.execution_context
        if context is None:
            return
        started_at = getattr(context, "_kaapi_db_started_at", None)
        duration_ms = (
            (time.perf_counter() - started_at) * 1000 if started_at is not None else 0.0
        )
        operation = getattr(context, "_kaapi_db_operation", None)
        record_db_query_finished(
            duration_ms=duration_ms, operation=operation, error=True
        )

    @event.listens_for(engine.pool, "checkout")
    def _on_checkout(dbapi_connection, connection_record, connection_proxy) -> None:
        del dbapi_connection, connection_record, connection_proxy
        _emit_pool_metrics(engine.pool)

    @event.listens_for(engine.pool, "checkin")
    def _on_checkin(dbapi_connection, connection_record) -> None:
        del dbapi_connection, connection_record
        _emit_pool_metrics(engine.pool)

    engine._kaapi_db_telemetry_instrumented = True
    logger.debug("[instrument_db_engine] SQLAlchemy DB telemetry enabled")
