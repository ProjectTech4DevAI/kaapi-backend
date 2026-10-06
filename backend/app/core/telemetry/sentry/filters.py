import re
from importlib import import_module
from typing import Any

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
    ("sentry_sdk.integrations.cohere", "CohereIntegration"),
    ("sentry_sdk.integrations.google_genai", "GoogleGenAIIntegration"),
    ("sentry_sdk.integrations.huggingface_hub", "HuggingfaceHubIntegration"),
    ("sentry_sdk.integrations.langchain", "LangchainIntegration"),
    ("sentry_sdk.integrations.langgraph", "LanggraphIntegration"),
    ("sentry_sdk.integrations.openai", "OpenAIIntegration"),
    ("sentry_sdk.integrations.openai_agents", "OpenAIAgentsIntegration"),
    ("sentry_sdk.integrations.pydantic_ai", "PydanticAIIntegration"),
)

_REDACTED = "[REDACTED]"

# LLM job kwargs land in Sentry via sentry_sdk's CeleryIntegration; these keys
# carry end-user text and must be redacted before that happens.
_LLM_JOB_TASK_NAMES = {
    "app.celery.tasks.job_execution.run_llm_job",
    "app.celery.tasks.job_execution.run_llm_chain_job",
    "app.celery.tasks.job_execution.run_response_job",
}
_SENSITIVE_REQUEST_DATA_KEYS = (
    "query",
    "request_metadata",
    "response",
    "output",
    "callback_url",
)


_SQL_OR_CONNECT = re.compile(r"^(select|insert|update|delete|connect)\b", re.IGNORECASE)
_HTTP_SEND_RECEIVE = re.compile(r"http (send|receive)$", re.IGNORECASE)
_DB_QUERY_SPAN = re.compile(r"^db\.query$", re.IGNORECASE)
_BARE_HTTP_METHOD = re.compile(
    r"^(GET|HEAD|OPTIONS|POST|PUT|PATCH|DELETE|TRACE|CONNECT)$", re.IGNORECASE
)
_NOISE_PATH = re.compile(
    r"(^/health/?$|^/robots\.txt$|^/favicon\.ico$|^/wp-admin|^/wp-login|^/xmlrpc\.php$)",
    re.IGNORECASE,
)


def _extract_path(event: Event) -> str:
    request = event.get("request")
    if not isinstance(request, dict):
        return ""

    url = request.get("url")
    if isinstance(url, str) and url:
        if "://" in url:
            after_scheme = url.split("://", 1)[1]
            if "/" in after_scheme:
                return "/" + after_scheme.split("/", 1)[1].split("?", 1)[0]
            return "/"
        return url.split("?", 1)[0]
    return ""


def _should_drop_transaction(event: Event) -> bool:
    transaction = str(event.get("transaction") or "").strip()
    path = _extract_path(event)

    # Sentry shows probe traffic as bare "GET"/"HEAD" transactions.
    if _BARE_HTTP_METHOD.match(transaction):
        return True

    if path and _NOISE_PATH.search(path):
        return True

    return False


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


def before_send_transaction_filter(event: Event, _hint: Hint) -> Event | None:
    """Drop ASGI/DB noise spans and scrub genai content before shipping."""
    if _should_drop_transaction(event):
        return None

    scrub_genai_content(event)

    spans = event.get("spans")
    if not isinstance(spans, list):
        return event

    filtered: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue

        data = span.get("data")
        if not isinstance(data, dict):
            data = {}
        desc = str(span.get("description") or span.get("name") or "").strip()
        op = str(span.get("op") or "").strip()

        if _HTTP_SEND_RECEIVE.search(desc):
            continue
        if _DB_QUERY_SPAN.search(desc) or _DB_QUERY_SPAN.search(op):
            continue
        if data.get("db.system") is not None:
            continue
        if _SQL_OR_CONNECT.match(desc):
            continue

        filtered.append(span)

    event["spans"] = filtered
    return event


def _scrub_request_pii(event: Event) -> None:
    request = event.get("request")
    if not isinstance(request, dict):
        return

    headers = request.get("headers")
    if isinstance(headers, dict):
        for key in list(headers):
            if str(key).lower() in _SENSITIVE_HEADERS:
                headers[key] = _REDACTED

    if "cookies" in request:
        request["cookies"] = _REDACTED
    if request.get("query_string"):
        request["query_string"] = _REDACTED


def _scrub_request_body(event: Event) -> None:
    """Drop the request body unconditionally — on LLM routes it is the user's message."""
    request = event.get("request")
    if isinstance(request, dict) and "data" in request:
        request["data"] = _REDACTED


def _redact_llm_job_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Replace end-user text and identifying URLs in request_data; keep config/ids."""
    request_data = kwargs.get("request_data")
    if not isinstance(request_data, dict):
        return kwargs

    redacted_request_data = dict(request_data)
    for key in _SENSITIVE_REQUEST_DATA_KEYS:
        if key in redacted_request_data:
            redacted_request_data[key] = _REDACTED

    redacted_kwargs = dict(kwargs)
    redacted_kwargs["request_data"] = redacted_request_data
    return redacted_kwargs


def _scrub_llm_job_kwargs(event: Event) -> None:
    """Strip end-user query/response text from LLM job celery-job context."""
    extra = event.get("extra")
    if not isinstance(extra, dict):
        return

    celery_job = extra.get("celery-job")
    if not isinstance(celery_job, dict):
        return

    if celery_job.get("task_name") not in _LLM_JOB_TASK_NAMES:
        return

    kwargs = celery_job.get("kwargs")
    if isinstance(kwargs, dict):
        celery_job["kwargs"] = _redact_llm_job_kwargs(kwargs)


def before_send_error_filter(event: Event, _hint: Hint) -> Event | None:
    """Drop probe/scanner error events; scrub genai content, request body, and PII."""
    try:
        if _should_drop_transaction(event):
            return None
        _scrub_request_body(event)
        _scrub_llm_job_kwargs(event)
        scrub_genai_content(event)
        if not settings.SENTRY_SEND_DEFAULT_PII:
            _scrub_request_pii(event)
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
