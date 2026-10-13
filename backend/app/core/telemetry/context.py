"""Request-scoped log context: log_context(), LogContextFilter, tenant tags and Sentry user binding."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum

import sentry_sdk
from opentelemetry import trace

logger = logging.getLogger(__name__)

_log_context_var: ContextVar[dict[str, str] | None] = ContextVar(
    "kaapi_log_context", default=None
)

# Sentry skips "_"-prefixed record attributes, so only LOG_MODULE_FIELD reaches Sentry Logs.
LOG_MODULE_FIELD = "log_module"
LOG_LOCATION_FIELD = "_log_location"


class LogModuleEnum(StrEnum):
    KNOWLEDGEBASE = "knowledgebase"
    LLM_CALL = "llm-call"
    LLM_CHAIN = "llm-chain"
    CONFIG_VERSION = "config-version"
    ASSESSMENT = "assessment"
    EVALS = "evals"
    FASTEVALS = "fastevals"


# Exact logger name (the module's __name__) -> owning module. A new logging file joins here.
_LOGGER_MODULES: dict[str, LogModuleEnum] = {
    # -------- KNOWLEDGEBASE --------
    "app.api.routes.collection_job": LogModuleEnum.KNOWLEDGEBASE,
    "app.api.routes.collections": LogModuleEnum.KNOWLEDGEBASE,
    "app.api.routes.doc_transformation_job": LogModuleEnum.KNOWLEDGEBASE,
    "app.api.routes.documents": LogModuleEnum.KNOWLEDGEBASE,
    "app.crud.collection.collection": LogModuleEnum.KNOWLEDGEBASE,
    "app.crud.collection.collection_job": LogModuleEnum.KNOWLEDGEBASE,
    "app.crud.document.doc_transformation_job": LogModuleEnum.KNOWLEDGEBASE,
    "app.crud.document.document": LogModuleEnum.KNOWLEDGEBASE,
    "app.crud.document_collection": LogModuleEnum.KNOWLEDGEBASE,
    "app.crud.rag.gemini": LogModuleEnum.KNOWLEDGEBASE,
    "app.crud.rag.open_ai": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.collections.create_collection": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.collections.delete_collection": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.collections.helpers": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.collections.providers.gemini": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.collections.providers.openai": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.collections.providers.registry": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.doctransform.job": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.doctransform.zerox_transformer": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.documents.helpers": LogModuleEnum.KNOWLEDGEBASE,
    "app.services.documents.validator": LogModuleEnum.KNOWLEDGEBASE,
    # -------- LLM_CALL --------
    "app.api.routes.llm": LogModuleEnum.LLM_CALL,
    "app.crud.llm": LogModuleEnum.LLM_CALL,
    "app.services.llm.guardrails": LogModuleEnum.LLM_CALL,
    "app.services.llm.jobs": LogModuleEnum.LLM_CALL,
    "app.services.llm.mappers": LogModuleEnum.LLM_CALL,
    "app.services.llm.providers.claude": LogModuleEnum.LLM_CALL,
    "app.services.llm.providers.eleven_ai": LogModuleEnum.LLM_CALL,
    "app.services.llm.providers.google_aistudio": LogModuleEnum.LLM_CALL,
    "app.services.llm.providers.google_gcp": LogModuleEnum.LLM_CALL,
    "app.services.llm.providers.open_ai": LogModuleEnum.LLM_CALL,
    "app.services.llm.providers.registry": LogModuleEnum.LLM_CALL,
    "app.services.llm.providers.sarvam_ai": LogModuleEnum.LLM_CALL,
    # -------- LLM_CHAIN --------
    "app.api.routes.llm_chain": LogModuleEnum.LLM_CHAIN,
    "app.crud.llm_chain": LogModuleEnum.LLM_CHAIN,
    "app.services.llm.chain.chain": LogModuleEnum.LLM_CHAIN,
    "app.services.llm.chain.executor": LogModuleEnum.LLM_CHAIN,
    # -------- CONFIG_VERSION --------
    "app.crud.config.config": LogModuleEnum.CONFIG_VERSION,
    "app.crud.config.version": LogModuleEnum.CONFIG_VERSION,
    # -------- ASSESSMENT --------
    "app.api.routes.assessment.api": LogModuleEnum.ASSESSMENT,
    "app.crud.assessment.api": LogModuleEnum.ASSESSMENT,
    "app.crud.assessment.submission": LogModuleEnum.ASSESSMENT,
    "app.services.assessment.api.batch": LogModuleEnum.ASSESSMENT,
    "app.services.assessment.api.callbacks": LogModuleEnum.ASSESSMENT,
    "app.services.assessment.api.result_files": LogModuleEnum.ASSESSMENT,
    "app.services.assessment.api.results": LogModuleEnum.ASSESSMENT,
    "app.services.assessment.api.submission": LogModuleEnum.ASSESSMENT,
    "app.services.assessment.api.submission_store": LogModuleEnum.ASSESSMENT,
    "app.services.assessment.submission": LogModuleEnum.ASSESSMENT,
    # -------- EVALS --------
    "app.api.routes.evaluations.v2.dataset": LogModuleEnum.EVALS,
    "app.api.routes.evaluations.v2.evaluation": LogModuleEnum.EVALS,
    "app.api.routes.evaluations.v2.iteration": LogModuleEnum.EVALS,
    "app.api.routes.evaluations.v2.prompt_improvement": LogModuleEnum.EVALS,
    "app.crud.evaluations.iteration": LogModuleEnum.EVALS,
    "app.crud.evaluations.judge": LogModuleEnum.EVALS,
    "app.crud.evaluations.judge_stage": LogModuleEnum.EVALS,
    "app.crud.evaluations.summary": LogModuleEnum.EVALS,
    "app.services.evaluations.iteration": LogModuleEnum.EVALS,
    "app.services.evaluations.iteration_checkpointer": LogModuleEnum.EVALS,
    "app.services.evaluations.iteration_graph": LogModuleEnum.EVALS,
    "app.services.evaluations.prompt_improvement": LogModuleEnum.EVALS,
    # -------- FASTEVALS --------
    "app.crud.evaluations.fast": LogModuleEnum.FASTEVALS,
    "app.crud.evaluations.fast_chunks": LogModuleEnum.FASTEVALS,
    "app.services.evaluations.fast": LogModuleEnum.FASTEVALS,
}


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
                sentry_sdk.set_tag("tenant.org_id", str(org_id))
            if project_id is not None:
                sentry_sdk.set_tag("tenant.project_id", str(project_id))
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
    *,
    tag: str | None = None,
    module: LogModuleEnum | None = None,
    **fields: str | int | float | bool | None,
) -> Generator[None]:
    """Attach structured log context for the current execution scope.

    `module` overrides the logger-path module for every log in the block, for flows
    that run another flow's code (e.g. an LLM chain calling LLM-call helpers).
    """
    current = _log_context_var.get() or {}
    payload = dict(current)
    if tag:
        payload["tag"] = tag
        payload.setdefault("system", tag)
    if module:
        payload[LOG_MODULE_FIELD] = module.value
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

        module = (context_payload or {}).get(LOG_MODULE_FIELD) or _LOGGER_MODULES.get(
            record.name
        )
        if module:
            setattr(record, LOG_MODULE_FIELD, str(module))
        setattr(
            record,
            LOG_LOCATION_FIELD,
            f"[{module}] {record.name}" if module else record.name,
        )

        if not hasattr(record, "lifecycle"):
            span = trace.get_current_span()
            if span is not None and span.is_recording():
                span_name = getattr(span, "name", None)
                if isinstance(span_name, str) and span_name:
                    record.lifecycle = span_name
        return True
