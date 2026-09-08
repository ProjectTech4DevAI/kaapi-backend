import re
from importlib import import_module
from typing import Any

from sentry_sdk.integrations import Integration

from app.core.config import settings

# Request headers stripped from error events while PII is off (case-insensitive).
_SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {"authorization", "cookie", "set-cookie", "x-api-key"}
)
_SCRUBBED = "[scrubbed]"

# Attribute keys carrying prompt/completion text; Langfuse is the only sink for those.
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

# event -> spans -> data -> unpacked message parts.
_GENAI_SCRUB_MAX_DEPTH: int = 8

# Auto-enabled per installed provider SDK; each records prompts once PII is on.
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


def _extract_path(event: dict[str, Any]) -> str:
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


def _should_drop_transaction(event: dict[str, Any]) -> bool:
    transaction = str(event.get("transaction") or "").strip()
    path = _extract_path(event)

    # Sentry shows probe traffic as bare "GET"/"HEAD" transactions.
    if _BARE_HTTP_METHOD.match(transaction):
        return True

    # Drop known non-app noise paths (probes / scanners).
    if path and _NOISE_PATH.search(path):
        return True

    return False


def genai_privacy_integrations() -> list[Integration]:
    """Sentry AI integrations pinned to include_prompts=False.

    Passed to `sentry_sdk.init(integrations=...)` so the explicit instance wins over the
    auto-enabled one. Absent provider SDKs are skipped: init raises on those.
    """
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


def scrub_genai_content(payload: Any, depth: int = 0) -> None:
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


def before_send_transaction_filter(
    event: dict[str, Any], hint: dict[str, Any]
) -> dict[str, Any] | None:
    """Drop low-signal spans and genai message content before they ship to Sentry.

    Filters out:
    - ASGI lifecycle spans ending with `http send` / `http receive`
    - DB spans carrying `db.system`
    - SQL / `connect` spans matched by description prefix
    - Custom DB query spans (`db.query`)
    - Prompt/completion attributes anywhere in the event (`scrub_genai_content`)
    """
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

        data = span.get("data") if isinstance(span.get("data"), dict) else {}
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


def _scrub_request_pii(event: dict[str, Any]) -> None:
    request = event.get("request")
    if not isinstance(request, dict):
        return

    headers = request.get("headers")
    if isinstance(headers, dict):
        for key in list(headers):
            if str(key).lower() in _SENSITIVE_HEADERS:
                headers[key] = _SCRUBBED

    if "cookies" in request:
        request["cookies"] = _SCRUBBED
    if request.get("query_string"):
        request["query_string"] = _SCRUBBED


def _scrub_request_body(event: dict[str, Any]) -> None:
    """Drop the request body unconditionally — on LLM routes it is the user's message."""
    request = event.get("request")
    if isinstance(request, dict) and "data" in request:
        request["data"] = _SCRUBBED


def before_send_error_filter(
    event: dict[str, Any], hint: dict[str, Any]
) -> dict[str, Any] | None:
    """Drop probe/scanner error events; scrub genai content, request body, and PII."""
    try:
        if _should_drop_transaction(event):
            return None
        _scrub_request_body(event)
        scrub_genai_content(event)
        if not settings.SENTRY_SEND_DEFAULT_PII:
            _scrub_request_pii(event)
    except Exception:
        return event
    return event


def before_send_log_filter(
    log: dict[str, Any], hint: dict[str, Any]
) -> dict[str, Any] | None:
    """Strip genai content from Sentry log attributes before shipping."""
    try:
        scrub_genai_content(log.get("attributes"))
    except Exception:
        return log
    return log
