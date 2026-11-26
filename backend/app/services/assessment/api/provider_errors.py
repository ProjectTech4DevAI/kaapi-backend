"""One-line, client-safe provider errors: ``[PROVIDER] code: message (request_id=...)``."""

import json
import re
from typing import Any

from app.models.llm.constants import Provider

# Signed attachment URLs carry credentials in the query string.
_URL_QUERY = re.compile(r"(https?://[^\s?\"'<>]+)\?[^\s\"'<>]*")

# google.rpc.Code numbers Gemini returns in per-row errors.
_GRPC_CODES = {
    1: "CANCELLED",
    2: "UNKNOWN",
    3: "INVALID_ARGUMENT",
    4: "DEADLINE_EXCEEDED",
    5: "NOT_FOUND",
    7: "PERMISSION_DENIED",
    8: "RESOURCE_EXHAUSTED",
    9: "FAILED_PRECONDITION",
    13: "INTERNAL",
    14: "UNAVAILABLE",
    16: "UNAUTHENTICATED",
}

_LABELS = {
    Provider.OPENAI: "OPENAI",
    Provider.ANTHROPIC: "ANTHROPIC",
    Provider.GOOGLE: "GEMINI",
    Provider.GOOGLE_AISTUDIO: "GEMINI",
    Provider.GOOGLE_GCP: "VERTEX",
}


# Distinct reasons named in a run-level message; each is shown in full, the rest counted.
MAX_LISTED_REASONS = 5


def join_reasons(reasons: list[str], separator: str = " | ") -> str:
    shown = reasons[:MAX_LISTED_REASONS]
    text = separator.join(shown)
    if len(reasons) > len(shown):
        text += f"{separator}+{len(reasons) - len(shown)} more distinct error(s)"
    return text


def provider_label(provider: str) -> str:
    return _LABELS.get(provider, provider.upper())


def redact_urls(text: str) -> str:
    return _URL_QUERY.sub(r"\1?<redacted>", text)


def format_provider_error(
    provider: str,
    *,
    code: Any = None,
    message: Any = None,
    request_id: Any = None,
    detail: str | None = None,
) -> str:
    code_text = str(code).strip() if code not in (None, "") else ""
    message_text = str(message).strip() if message not in (None, "") else ""
    body = (
        f"{code_text}: {message_text}"
        if code_text and message_text
        else message_text or code_text or "unknown error"
    )
    text = f"[{provider_label(provider)}] {body}"
    if detail:
        text += f" ({detail})"
    if request_id:
        text += f" (request_id={request_id})"
    return redact_urls(text)


def error_from_payload(provider: str, payload: Any) -> str:
    """Format a provider error given as a string, a dict, or an envelope around one."""
    if payload is None:
        return format_provider_error(provider)
    if isinstance(payload, str):
        label = f"[{provider_label(provider)}]"
        return (
            redact_urls(payload)
            if payload.startswith(label)
            else (format_provider_error(provider, message=payload))
        )
    if not isinstance(payload, dict):
        return format_provider_error(provider, message=str(payload))

    request_id = payload.get("request_id")
    inner = payload.get("error")
    if isinstance(inner, str) and inner:
        return format_provider_error(provider, message=inner, request_id=request_id)
    if isinstance(inner, dict):
        payload = inner
    raw_code = payload.get("code")
    # Most specific first: Google status, OpenAI string code, Anthropic/OpenAI type.
    code = (
        payload.get("status")
        or (raw_code if isinstance(raw_code, str) and raw_code else None)
        or payload.get("type")
        or raw_code
    )
    if isinstance(code, int):
        code = _GRPC_CODES.get(code, code)
    message = payload.get("message")
    if message is None:
        message = json.dumps(payload, default=str)
    detail = f"param={payload['param']}" if payload.get("param") else None
    return format_provider_error(
        provider, code=code, message=message, request_id=request_id, detail=detail
    )


def error_from_vertex_status(provider: str, status: Any) -> str:
    """Vertex output lines put the error in a string like ``Bad Request: {json}``."""
    if not isinstance(status, str):
        return error_from_payload(provider, status)
    _, sep, raw = status.partition(": ")
    if sep:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return error_from_payload(provider, parsed)
    return format_provider_error(provider, message=status)


def row_error(provider: str, raw: Any) -> str:
    """A batch row's error: the adapter's structured ``error_raw`` or its plain ``error``."""
    if provider == Provider.GOOGLE_GCP and isinstance(raw, str):
        return error_from_vertex_status(provider, raw)
    return error_from_payload(provider, raw)


def job_failure_reason(provider: str, status_result: dict[str, Any]) -> str | None:
    """Why a whole batch failed, from the keys an adapter's get_batch_status adds."""
    errors = status_result.get("provider_errors")
    if errors:
        messages = [
            format_provider_error(
                provider,
                code=e.get("code"),
                message=e.get("message"),
                detail=", ".join(
                    part
                    for part in (
                        f"param={e['param']}" if e.get("param") else "",
                        f"line={e['line']}" if e.get("line") is not None else "",
                    )
                    if part
                )
                or None,
            )
            for e in errors
        ]
        return join_reasons(list(dict.fromkeys(messages)), separator="; ")
    if status_result.get("provider_error"):
        return error_from_payload(provider, status_result["provider_error"])
    if status_result.get("error_message"):
        return format_provider_error(provider, message=status_result["error_message"])
    return None


_SDK_MODULES = frozenset({"anthropic", "openai", "google", "httpx", "httpx2"})


def describe_exception(provider: str, exc: BaseException) -> str:
    """Client-facing text for an exception raised while talking to a provider or bucket."""
    # Wrappers (e.g. CloudStorageError) keep the real error as their cause.
    root = exc.__cause__ if exc.__cause__ is not None else exc

    if hasattr(root, "status_code"):  # anthropic / openai SDKs
        body = getattr(root, "body", None)
        # Anthropic wraps the error object in {"error": ...}; OpenAI passes it bare.
        error = body.get("error", body) if isinstance(body, dict) else None
        if isinstance(error, dict) and error.get("message"):
            return error_from_payload(
                provider, {**error, "request_id": getattr(root, "request_id", None)}
            )
        # FastAPI HTTPException (e.g. missing credentials) carries its text in .detail.
        message = getattr(root, "message", None) or getattr(root, "detail", None)
        return format_provider_error(
            provider, code=root.status_code, message=message or str(root)
        )

    status = getattr(root, "status", None)
    code = getattr(root, "code", None)
    message = getattr(root, "message", None)
    if message is not None and (status or code):  # google-genai / google-api-core
        first = (getattr(root, "errors", None) or [None])[0]
        if isinstance(first, dict) and first.get("message"):
            return format_provider_error(
                provider,
                code=first.get("reason") or status or code,
                message=first["message"],
            )
        return format_provider_error(provider, code=status or code, message=message)

    if type(root).__module__.split(".")[0] in _SDK_MODULES:
        return format_provider_error(
            provider, code=type(root).__name__, message=str(root) or None
        )
    # May hold internal detail (paths, SQL); logged server-side only.
    return format_provider_error(
        provider,
        code="internal_error",
        message="the batch could not be submitted because of an internal error",
    )
