"""Sentry before_send filters: drop probe events, scrub genai content, request PII and LLM job kwargs."""

import re
from importlib import import_module
from urllib.parse import urlsplit

from sentry_sdk.integrations import Integration
from sentry_sdk.types import Event, Hint, Log

from app.core.config import settings

_SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {"authorization", "cookie", "set-cookie", "x-api-key"}
)

_GENAI_CONTENT_KEYS: frozenset[str] = frozenset(
    {
        "gen_ai.request.messages",
        "gen_ai.response.text",
        "gen_ai.response.tool_calls",
        "gen_ai.choice",
        "gen_ai.embeddings.input",
        "gen_ai.system_instructions",
        "gen_ai.tool.input",
        "gen_ai.tool.output",
        "gen_ai.user.message",
        "gen_ai.prompt",
        "gen_ai.completion",
        "ai.input_messages",
        "ai.responses",
        "ai.prompt",
        "ai.texts",
        "ai.tool_calls",
        "ai.function_call",
        "ai.search_queries",
        "langfuse.observation.input",
        "langfuse.observation.output",
        "langfuse.trace.input",
        "langfuse.trace.output",
    }
)

_GENAI_SCRUB_MAX_DEPTH: int = 8

# Sentry auto-enables these per installed SDK; they record prompts when PII is on.
_GENAI_INTEGRATIONS: tuple[tuple[str, str], ...] = (
    ("sentry_sdk.integrations.anthropic", "AnthropicIntegration"),
    ("sentry_sdk.integrations.google_genai", "GoogleGenAIIntegration"),
    ("sentry_sdk.integrations.langgraph", "LanggraphIntegration"),
    ("sentry_sdk.integrations.openai", "OpenAIIntegration"),
)

_REDACTED = "[REDACTED]"

# CeleryIntegration ships task kwargs; these keys carry end-user text.
_LLM_JOB_TASK_NAMES = {
    "app.celery.tasks.job_execution.run_llm_job",
    "app.celery.tasks.job_execution.run_llm_chain_job",
}
_SENSITIVE_REQUEST_DATA_KEYS = ("query", "request_metadata", "callback_url")


_BARE_HTTP_METHOD = re.compile(
    r"^(GET|HEAD|OPTIONS|POST|PUT|PATCH|DELETE|TRACE|CONNECT)$", re.IGNORECASE
)
_NOISE_PATH = re.compile(
    r"(^/health/?$|^/robots\.txt$|^/favicon\.ico$|^/wp-admin|^/wp-login|^/xmlrpc\.php$)",
    re.IGNORECASE,
)


def _request_path(event: Event) -> str:
    request = event.get("request")
    url = request.get("url") if isinstance(request, dict) else None
    return urlsplit(url).path if isinstance(url, str) else ""


def _is_probe_event(event: Event) -> bool:
    # Sentry shows probe traffic as bare "GET"/"HEAD" transactions.
    transaction = str(event.get("transaction") or "").strip()
    if _BARE_HTTP_METHOD.match(transaction):
        return True
    path = _request_path(event)
    return bool(path and _NOISE_PATH.search(path))


def genai_privacy_integrations() -> list[Integration]:
    """AI integrations pinned to include_prompts=False; missing SDKs are skipped."""
    integrations: list[Integration] = []
    for module_path, class_name in _GENAI_INTEGRATIONS:
        try:
            integration_class = getattr(import_module(module_path), class_name)
            integrations.append(integration_class(include_prompts=False))
        except Exception:
            continue
    return integrations


def _is_genai_content_key(key: object) -> bool:
    """True for a genai content key or one of its unpacked sub-keys (`...messages.0.content`)."""
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    return any(
        lowered == content_key or lowered.startswith(f"{content_key}.")
        for content_key in _GENAI_CONTENT_KEYS
    )


def scrub_genai_content(payload: object, depth: int = 0) -> None:
    """Strip genai prompt/completion values in-place from a nested Sentry payload."""
    if depth > _GENAI_SCRUB_MAX_DEPTH:
        return
    if isinstance(payload, dict):
        for key in list(payload):
            if _is_genai_content_key(key):
                del payload[key]
            else:
                scrub_genai_content(payload[key], depth + 1)
    elif isinstance(payload, list):
        for item in payload:
            scrub_genai_content(item, depth + 1)


def _scrub_request(event: Event) -> None:
    """Drop the body always (LLM routes carry the user message); headers/cookies/query only when PII is off."""
    request = event.get("request")
    if not isinstance(request, dict):
        return
    if "data" in request:
        request["data"] = _REDACTED
    if settings.SENTRY_SEND_DEFAULT_PII:
        return
    headers = request.get("headers")
    if isinstance(headers, dict):
        for key in headers:
            if str(key).lower() in _SENSITIVE_HEADERS:
                headers[key] = _REDACTED
    if "cookies" in request:
        request["cookies"] = _REDACTED
    if request.get("query_string"):
        request["query_string"] = _REDACTED


def _scrub_llm_job_kwargs(event: Event) -> None:
    """Redact end-user text and callback URL in the celery-job kwargs of LLM tasks."""
    extra = event.get("extra")
    celery_job = extra.get("celery-job") if isinstance(extra, dict) else None
    if not isinstance(celery_job, dict):
        return
    if celery_job.get("task_name") not in _LLM_JOB_TASK_NAMES:
        return
    kwargs = celery_job.get("kwargs")
    request_data = kwargs.get("request_data") if isinstance(kwargs, dict) else None
    if not isinstance(request_data, dict):
        return
    for key in _SENSITIVE_REQUEST_DATA_KEYS:
        if key in request_data:
            request_data[key] = _REDACTED


def before_send_transaction_filter(event: Event, _hint: Hint) -> Event | None:
    """Drop probe transactions and scrub genai content before shipping."""
    if _is_probe_event(event):
        return None
    scrub_genai_content(event)
    return event


def before_send_error_filter(event: Event, _hint: Hint) -> Event | None:
    """Drop probe events; scrub request body/PII, LLM job kwargs and genai content."""
    try:
        if _is_probe_event(event):
            return None
        _scrub_request(event)
        _scrub_llm_job_kwargs(event)
        scrub_genai_content(event)
    except Exception:
        return event
    return event


def before_send_log_filter(log: Log, _hint: Hint) -> Log | None:
    """Strip genai content from Sentry log attributes before shipping."""
    try:
        scrub_genai_content(log.get("attributes"))
    except Exception:
        return log
    return log
