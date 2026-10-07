"""Request-scoped log context: log_context(), LogContextFilter, tenant tags and Sentry user binding."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

import sentry_sdk
from opentelemetry import trace

logger = logging.getLogger(__name__)

_log_context_var: ContextVar[dict[str, str] | None] = ContextVar(
    "kaapi_log_context", default=None
)


def set_request_log_context(
    org_id: int | None = None,
    project_id: int | None = None,
) -> None:
    """Attach org/project to the request's log context and Sentry tags."""
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
                sentry_sdk.set_tag("kaapi.organization_id", str(org_id))
            if project_id is not None:
                sentry_sdk.set_tag("kaapi.project_id", str(project_id))
    except Exception:
        logger.debug("[set_request_log_context] Failed to tag Sentry scope")


def bind_sentry_user(
    user_id: int | None = None,
    org_id: int | None = None,
    project_id: int | None = None,
) -> None:
    """Bind caller ids to the Sentry scope only; log context would add per-user cardinality."""
    try:
        if not sentry_sdk.get_client().is_active():
            return
        sentry_user: dict[str, str] = {}
        if user_id is not None:
            sentry_user["id"] = str(user_id)
        if org_id is not None:
            sentry_user["org_id"] = str(org_id)
        if project_id is not None:
            sentry_user["project_id"] = str(project_id)
        if sentry_user:
            sentry_sdk.set_user(sentry_user)
    except Exception:
        logger.debug("[bind_sentry_user] Failed to bind Sentry user")


@contextmanager
def log_context(
    *, tag: str | None = None, **fields: str | int | float | bool | None
) -> Generator[None]:
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
