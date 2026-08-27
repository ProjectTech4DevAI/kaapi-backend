"""Unit tests for get_langfuse_tracer_provider.

Langfuse spans must not land on the global OTel provider — Sentry's SpanProcessor
lives there and re-exports every span as a duplicate root trace.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from app.core.langfuse.langfuse import get_langfuse_tracer_provider


def test_returns_sdk_tracer_provider() -> None:
    assert isinstance(get_langfuse_tracer_provider(), TracerProvider)


def test_is_cached_across_calls() -> None:
    assert get_langfuse_tracer_provider() is get_langfuse_tracer_provider()


def test_is_not_the_global_provider() -> None:
    assert get_langfuse_tracer_provider() is not trace.get_tracer_provider()
