"""LLM call telemetry: gen_ai span attributes, call/token metrics, HTTP span suppression around provider calls."""

import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.instrumentation.utils import (
    suppress_http_instrumentation as otel_suppress_http_instrumentation,
)

from app.core.telemetry.constants import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_AVAILABLE_TOOLS,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    TENANT_ORG_ID,
    TENANT_PROJECT_ID,
)
from app.core.telemetry.metrics import emit_sentry_metric

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
        GEN_AI_SYSTEM: provider,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_OPERATION_NAME: operation,
    }
    if organization_id is not None:
        attrs[TENANT_ORG_ID] = str(organization_id)
    if project_id is not None:
        attrs[TENANT_PROJECT_ID] = str(project_id)
    return attrs


def record_llm_call_started(
    provider: str,
    model: str,
    operation: str,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> None:
    """Emit LLM call-start metric to Sentry."""
    attrs = _llm_call_attrs(provider, model, operation, organization_id, project_id)
    emit_sentry_metric("count", "llm.call.count", 1, attributes=attrs)


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
    attrs = _llm_call_attrs(provider, model, operation, organization_id, project_id)

    emit_sentry_metric(
        "distribution",
        "llm.call.duration",
        duration_ms,
        unit="millisecond",
        attributes=attrs,
    )
    if error:
        emit_sentry_metric("count", "llm.call.error", 1, attributes=attrs)
    if input_tokens is not None:
        emit_sentry_metric("count", "llm.tokens.input", input_tokens, attributes=attrs)
    if output_tokens is not None:
        emit_sentry_metric(
            "count", "llm.tokens.output", output_tokens, attributes=attrs
        )
    if total_tokens is not None:
        emit_sentry_metric("count", "llm.tokens.total", total_tokens, attributes=attrs)


def set_gen_ai_request_attributes(
    span: trace.Span,
    *,
    provider: str,
    model: str,
    operation: str,
    organization_id: int | None,
    project_id: int | None,
    params: Mapping[str, object] | None = None,
) -> None:
    """Set OTel GenAI request attributes on `span` (semantic-convention keys + tenant ids)."""
    span.set_attribute(GEN_AI_SYSTEM, provider)
    span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
    span.set_attribute(GEN_AI_OPERATION_NAME, operation)
    if model:
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
    if organization_id is not None:
        span.set_attribute(TENANT_ORG_ID, str(organization_id))
    if project_id is not None:
        span.set_attribute(TENANT_PROJECT_ID, str(project_id))

    params = params or {}
    for attr_key, param_key in (
        (GEN_AI_REQUEST_TEMPERATURE, "temperature"),
        (GEN_AI_REQUEST_MAX_TOKENS, "max_tokens"),
        (GEN_AI_REQUEST_TOP_P, "top_p"),
        (GEN_AI_REQUEST_PRESENCE_PENALTY, "presence_penalty"),
        (GEN_AI_REQUEST_FREQUENCY_PENALTY, "frequency_penalty"),
    ):
        value = params.get(param_key)
        if isinstance(value, str | bool | int | float):
            span.set_attribute(attr_key, value)

    tools = params.get("tools")
    if tools is not None:
        span.set_attribute(GEN_AI_REQUEST_AVAILABLE_TOOLS, json.dumps(tools))


def set_gen_ai_response_attributes(
    span: trace.Span, *, response: "LLMCallResponse"
) -> None:
    """Set OTel GenAI response attributes (usage, model) on `span`."""
    usage = response.usage
    if usage:
        reasoning_tokens = usage.reasoning_tokens or 0
        output_tokens = usage.output_tokens
        if reasoning_tokens and usage.input_tokens + output_tokens < usage.total_tokens:
            output_tokens += reasoning_tokens
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        span.set_attribute(GEN_AI_USAGE_TOTAL_TOKENS, usage.total_tokens)
        if reasoning_tokens:
            span.set_attribute(GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, reasoning_tokens)

    if response.response and response.response.model:
        span.set_attribute(GEN_AI_RESPONSE_MODEL, response.response.model)


@contextmanager
def suppress_http_instrumentation() -> Generator[None]:
    """Skip OTel HTTP client spans for the block; the LLM span already covers the call."""
    with otel_suppress_http_instrumentation():
        yield
