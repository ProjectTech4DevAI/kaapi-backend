"""Observability package: public API for OTel tracing, Sentry init, log context, HTTP/DB/LLM metrics."""

from app.core.telemetry.context import (
    bind_sentry_user,
    log_context,
    set_request_log_context,
)
from app.core.telemetry.db import instrument_db_engine
from app.core.telemetry.http import record_http_request, record_unmatched_request
from app.core.telemetry.llm import (
    record_llm_call_finished,
    record_llm_call_started,
    set_gen_ai_request_attributes,
    set_gen_ai_response_attributes,
    suppress_http_instrumentation,
)
from app.core.telemetry.metrics import record_rate_threshold, record_stale_pending_jobs
from app.core.telemetry.sentry import init_sentry
from app.core.telemetry.tracing import flush_telemetry, instrument_app, setup_telemetry

__all__ = [
    "bind_sentry_user",
    "flush_telemetry",
    "init_sentry",
    "instrument_app",
    "instrument_db_engine",
    "log_context",
    "record_http_request",
    "record_llm_call_finished",
    "record_unmatched_request",
    "record_llm_call_started",
    "record_rate_threshold",
    "record_stale_pending_jobs",
    "set_gen_ai_request_attributes",
    "set_gen_ai_response_attributes",
    "set_request_log_context",
    "setup_telemetry",
    "suppress_http_instrumentation",
]
