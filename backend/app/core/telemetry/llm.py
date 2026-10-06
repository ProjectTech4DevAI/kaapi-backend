import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.instrumentation.utils import _SUPPRESS_HTTP_INSTRUMENTATION_KEY

from app.core.config import settings
from app.core.telemetry.metrics import _emit_sentry_metric

if TYPE_CHECKING:
    from app.models.llm.response import LLMCallResponse


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
def suppress_http_instrumentation() -> Generator[None]:
    """Skip OTel HTTP client spans for the block; the LLM span already covers the call."""
    token = otel_context.attach(
        otel_context.set_value(_SUPPRESS_HTTP_INSTRUMENTATION_KEY, True)
    )
    try:
        yield
    finally:
        otel_context.detach(token)
