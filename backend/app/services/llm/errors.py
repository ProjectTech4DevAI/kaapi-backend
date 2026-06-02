"""Shared error-capture plumbing for LLM provider calls.

Providers catch typed upstream exceptions (e.g. `openai.OpenAIError`) inside
their `execute` methods and only return a string back to the caller. The
caller has no way to recover the upstream HTTP status or error code from
that string. To avoid changing every provider's return signature, providers
populate this context-var at the catch site; `execute_llm_call` reads it
after the provider call and folds the metadata into `BlockResult.error_detail`.

Providers that don't populate the var work fine — their `error_detail` will
just have the message + the request-side fields (`conversation_id`, `provider`).
"""

from contextvars import ContextVar
from typing import TypedDict


class ProviderErrorMeta(TypedDict, total=False):
    provider_status_code: int
    error_type: str  # rate_limit | authentication | invalid_request | timeout | provider_error


_provider_error_meta: ContextVar[ProviderErrorMeta | None] = ContextVar(
    "kaapi_provider_error_meta", default=None
)


def set_provider_error_meta(meta: ProviderErrorMeta) -> None:
    """Called by a provider's `execute` when it catches a typed upstream error."""
    _provider_error_meta.set(meta)


def consume_provider_error_meta() -> ProviderErrorMeta | None:
    """Read and clear the meta set by a provider call. Returns None if nothing was set."""
    meta = _provider_error_meta.get()
    _provider_error_meta.set(None)
    return meta
