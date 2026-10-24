import json
import logging
import threading
import uuid
from collections.abc import Callable
from functools import wraps
from typing import Any

from asgi_correlation_id import correlation_id
from langfuse import Langfuse, LangfuseOtelSpanAttributes
from langfuse._client.span import LangfuseGeneration, LangfuseSpan
from langfuse.api.core.api_error import ApiError
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider

from app.core.config import settings
from app.core.telemetry.constants import GEN_AI_RESPONSE_ID, LLM_COMPLETION_TYPE
from app.core.telemetry.llm import billable_output_tokens
from app.models.llm import (
    AudioContent,
    AudioOutput,
    ImageContent,
    LLMCallResponse,
    NativeCompletionConfig,
    PDFContent,
    QueryParams,
    TextInput,
    TextOutput,
)
from app.models.llm.request import QueryInput

TraceInput = dict[str, object] | list[dict[str, object]]
ModelParameterValue = str | int | float | bool | None

# Prompt text belongs in the generation input, not in model parameters.
_PROMPT_PARAM_KEYS = frozenset(
    {"model", "instructions", "input", "messages", "system", "director_notes"}
)

logger = logging.getLogger(__name__)

_LANGFUSE_TRACER_PROVIDER: TracerProvider | None = None
_LANGFUSE_TRACER_PROVIDER_LOCK = threading.Lock()


def get_langfuse_tracer_provider() -> TracerProvider:
    """Isolated OTel provider for Langfuse so LLM spans never reach Sentry.

    Sentry's SpanProcessor sits on the global provider and exports every span
    with no per-project filter; sharing it surfaces each Langfuse span in Sentry
    as a duplicate root trace. Langfuse's own processors filter by public_key, so
    one shared isolated provider stays multi-tenant safe.
    """
    global _LANGFUSE_TRACER_PROVIDER
    if _LANGFUSE_TRACER_PROVIDER is None:
        with _LANGFUSE_TRACER_PROVIDER_LOCK:
            if _LANGFUSE_TRACER_PROVIDER is None:
                _LANGFUSE_TRACER_PROVIDER = TracerProvider(
                    resource=Resource.create({SERVICE_NAME: settings.OTEL_SERVICE_NAME})
                )
    return _LANGFUSE_TRACER_PROVIDER


def format_langfuse_error(exc: Exception) -> str:
    """Return a concise message for a Langfuse SDK exception.

    Langfuse 4.x's ``ApiError.__str__`` dumps the full HTTP response including
    every response header, e.g.::

        headers: {'date': ..., 'etag': ...}, status_code: 404, body: {...}

    For user-facing/log error messages we only want the actionable parts, so we
    drop the headers and keep ``status_code`` and ``body`` (matching the 2.x
    string format). Non-API exceptions fall back to ``str(exc)``.
    """
    if isinstance(exc, ApiError):
        return f"status_code: {exc.status_code}, body: {exc.body}"
    return str(exc)


def _serialize(value: Any) -> str:
    """Serialize a trace attribute value to a JSON string for OTel span attributes."""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _to_usage_details(usage: dict[str, Any] | None) -> dict[str, int] | None:
    """Convert a v2-style usage dict to the v4 ``usage_details`` (int-only) format.

    v2 callers pass ``{"input", "output", "total", "unit": "TOKENS"}``; v4 expects
    ``usage_details: dict[str, int]`` and has no ``unit`` field, so non-int values
    (e.g. ``"TOKENS"``) are dropped.
    """
    if not usage:
        return None
    return {k: v for k, v in usage.items() if isinstance(v, int)}


def set_trace_attributes(
    span: LangfuseSpan,
    *,
    name: str | None = None,
    input: Any = None,
    output: Any = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Set trace-level attributes on a root span's underlying OTel span.

    v4 has no public imperative setter for trace session_id/tags/metadata (it steers
    callers to the ``propagate_attributes`` context manager, which doesn't fit an
    imperative, method-by-method tracing lifecycle), so we write the public OTel
    attribute keys (``LangfuseOtelSpanAttributes.TRACE_*``) directly on the root span.
    Trace metadata round-trips per-key as ``langfuse.trace.metadata.<key>``.
    """
    attrs: dict[str, Any] = {}
    if name is not None:
        attrs[LangfuseOtelSpanAttributes.TRACE_NAME] = name
    if input is not None:
        attrs[LangfuseOtelSpanAttributes.TRACE_INPUT] = _serialize(input)
    if output is not None:
        attrs[LangfuseOtelSpanAttributes.TRACE_OUTPUT] = _serialize(output)
    if session_id is not None:
        attrs[LangfuseOtelSpanAttributes.TRACE_SESSION_ID] = session_id
    if tags:
        attrs[LangfuseOtelSpanAttributes.TRACE_TAGS] = tags
    if metadata:
        for key, value in metadata.items():
            attrs[f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{key}"] = (
                value if isinstance(value, str) else _serialize(value)
            )
    if attrs:
        span._otel_span.set_attributes(attrs)


def _media_content_metadata(
    content: AudioContent | ImageContent | PDFContent,
) -> dict[str, object]:
    """Media fields without the base64 or URL payload, which is large and may be private."""
    return {
        "format": content.format,
        "mime_type": content.mime_type,
        "length": len(content.value),
    }


def _query_input_item(query_input: str | QueryInput) -> dict[str, object]:
    if isinstance(query_input, str):
        return {"type": "text", "content": query_input, "format": "text"}
    if isinstance(query_input, TextInput):
        return {
            "type": "text",
            "content": query_input.content.value,
            "format": query_input.content.format,
            "language_code": query_input.content.language_code,
        }
    contents = (
        query_input.content
        if isinstance(query_input.content, list)
        else [query_input.content]
    )
    media = [_media_content_metadata(content) for content in contents]
    if len(media) == 1:
        return {"type": query_input.type, **media[0]}
    return {"type": query_input.type, "items": media}


def extract_query_input(
    query_input: str | QueryInput | list[QueryInput],
) -> TraceInput:
    """Structured trace input (type, format, language code) instead of the Pydantic repr.

    Text keeps its content; audio, image and PDF send metadata only, like extract_response_output.
    """
    if isinstance(query_input, list):
        return [_query_input_item(item) for item in query_input]
    return _query_input_item(query_input)


def _input_metadata(trace_input: TraceInput) -> dict[str, str]:
    """Input type, format and language code as filterable metadata."""
    items = trace_input if isinstance(trace_input, list) else [trace_input]
    types: dict[str, None] = {}
    formats: dict[str, None] = {}
    languages: dict[str, None] = {}
    for item in items:
        types[str(item["type"])] = None
        parts = item.get("items")
        for part in parts if isinstance(parts, list) else [item]:
            if part.get("format"):
                formats[str(part["format"])] = None
        if item.get("language_code"):
            languages[str(item["language_code"])] = None
    metadata = {
        "llm.input.type": ",".join(types),
        "llm.input.format": ",".join(formats),
    }
    if languages:
        metadata["llm.input.language_code"] = ",".join(languages)
    return metadata


def _output_metadata(llm_output: TextOutput | AudioOutput | None) -> dict[str, str]:
    if isinstance(llm_output, TextOutput):
        metadata = {
            "llm.output.type": "text",
            "llm.output.format": llm_output.content.format,
        }
        if llm_output.content.language_code:
            metadata["llm.output.language_code"] = llm_output.content.language_code
        return metadata
    if isinstance(llm_output, AudioOutput):
        return {
            "llm.output.type": "audio",
            "llm.output.format": llm_output.content.format,
        }
    return {}


def _model_parameters(params: dict[str, object]) -> dict[str, ModelParameterValue]:
    """Scalar config params, one nested level flattened (``reasoning.effort``)."""
    scalars = (str, int, float, bool)
    result: dict[str, ModelParameterValue] = {}
    for key, value in params.items():
        if key in _PROMPT_PARAM_KEYS:
            continue
        if isinstance(value, scalars):
            result[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key not in _PROMPT_PARAM_KEYS and isinstance(sub_value, scalars):
                    result[f"{key}.{sub_key}"] = sub_value
    return result


def _generation_input(params: dict[str, object], trace_input: TraceInput) -> object:
    """Chat-format input so Langfuse shows the system prompt."""
    instructions = params.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        return trace_input
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": trace_input},
    ]


def extract_response_output(
    llm_output: TextOutput | AudioOutput | None,
) -> str | dict[str, Any]:
    """Extract output value from LLM output for logging/tracing.

    Args:
        llm_output: The output (TextOutput, AudioOutput, or None)

    Returns:
        String value for text output, or dict with metadata for audio output
    """
    if not llm_output:
        return ""

    if isinstance(llm_output, TextOutput):
        return llm_output.content.value
    elif isinstance(llm_output, AudioOutput):
        # For audio, return metadata instead of the full base64 data
        return {
            "type": "audio",
            "format": llm_output.content.format,
            "mime_type": llm_output.content.mime_type,
            "length": len(llm_output.content.value),
        }
    return str(llm_output)


class LangfuseTracer:
    def __init__(
        self,
        credentials: dict | None = None,
        session_id: str | None = None,
        response_id: str | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.langfuse: Langfuse | None = None
        self.trace: LangfuseSpan | None = None
        self.generation: LangfuseGeneration | None = None
        self._failed = False

        has_credentials = (
            credentials
            and "public_key" in credentials
            and "secret_key" in credentials
            and "host" in credentials
        )

        if has_credentials:
            try:
                # v4 caches one client per public_key (thread-safe singleton); passing
                # explicit creds keeps this multi-tenant. Never use bare get_client() —
                # with multiple projects in-process it returns a disabled no-op client.
                self.langfuse = Langfuse(
                    public_key=credentials["public_key"],
                    secret_key=credentials["secret_key"],
                    host=credentials["host"],
                    tracing_enabled=True,  # This ensures the client is active
                    environment=settings.ENVIRONMENT,
                    tracer_provider=get_langfuse_tracer_provider(),
                )
            except Exception as e:
                logger.warning(
                    f"[LangfuseTracer] Failed to initialize: {format_langfuse_error(e)}"
                )
                self._failed = True
                return

            if response_id:
                try:
                    traces = self.langfuse.api.trace.list(tags=response_id).data
                    if traces:
                        self.session_id = traces[0].session_id
                except Exception as e:
                    logger.debug(
                        f"[LangfuseTracer] Session resume failed: {format_langfuse_error(e)}"
                    )

            logger.info(
                f"[LangfuseTracer] Tracing enabled | session_id={self.session_id}"
            )
        else:
            logger.warning("[LangfuseTracer] Tracing disabled - missing credentials")

    def _langfuse_call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if self._failed:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"[LangfuseTracer] {getattr(fn, '__name__', 'operation')} failed: {format_langfuse_error(e)}"
            )
            self._failed = True
            return None

    def start_trace(
        self,
        name: str,
        input: Any,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        if self._failed or not self.langfuse:
            return
        metadata = metadata or {}
        metadata["request_id"] = correlation_id.get() or "N/A"
        self.trace = self._langfuse_call(
            self.langfuse.start_observation,
            as_type="span",
            name=name,
            input=input,
            metadata=metadata,
        )
        if self.trace:
            self._langfuse_call(
                set_trace_attributes,
                self.trace,
                name=name,
                input=input,
                session_id=self.session_id,
                tags=tags,
            )

    def start_generation(
        self,
        name: str,
        input: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._failed or not self.trace:
            return
        self.generation = self._langfuse_call(
            self.trace.start_observation,
            as_type="generation",
            name=name,
            input=input,
            metadata=metadata or {},
        )

    def end_generation(
        self,
        output: dict[str, Any],
        usage: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> None:
        if self._failed or not self.generation:
            return
        self._langfuse_call(
            self.generation.update,
            output=output,
            usage_details=_to_usage_details(usage),
            model=model,
        )
        self._langfuse_call(self.generation.end)

    def update_trace(self, tags: list[str], output: dict[str, Any]) -> None:
        if self._failed or not self.trace:
            return
        self._langfuse_call(set_trace_attributes, self.trace, tags=tags, output=output)
        self._langfuse_call(self.trace.end)

    def log_error(self, error_message: str, response_id: str | None = None) -> None:
        if self._failed:
            return
        if self.generation:
            self._langfuse_call(self.generation.update, output={"error": error_message})
            self._langfuse_call(self.generation.end)
        if self.trace:
            self._langfuse_call(
                set_trace_attributes,
                self.trace,
                tags=[response_id] if response_id else None,
                output={"status": "failure", "error": error_message},
            )
            self._langfuse_call(self.trace.end)

    def flush(self) -> None:
        if self._failed or not self.langfuse:
            return
        self._langfuse_call(self.langfuse.flush)


def observe_llm_execution(
    session_id: str | None = None,
    credentials: dict | None = None,
    trace_metadata: dict[str, str] | None = None,
) -> Callable:
    """Decorator to add Langfuse observability to LLM provider execute methods.

    Args:
        credentials: Langfuse credentials with public_key, secret_key, and host
        session_id: Session ID for grouping traces (conversation_id)
        trace_metadata: Call IDs, same keys as the Sentry spans (_CallContext.identity_attributes)

    Usage:
        decorated_execute = observe_llm_execution(
            credentials=langfuse_creds,
            session_id=conversation_id
        )(provider_instance.execute)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            completion_config: NativeCompletionConfig, query: QueryParams, **kwargs
        ):
            if not credentials:
                return func(completion_config, query, **kwargs)

            try:
                # See LangfuseTracer.__init__: explicit per-key client keeps this
                # multi-tenant under v4's public_key-keyed client cache.
                langfuse = Langfuse(
                    public_key=credentials.get("public_key"),
                    secret_key=credentials.get("secret_key"),
                    host=credentials.get("host"),
                    environment=settings.ENVIRONMENT,
                    tracer_provider=get_langfuse_tracer_provider(),
                )
                logger.info(
                    f"[observe_llm_execution] Tracing enabled | session_id={session_id or 'auto'}"
                )
            except Exception as e:
                logger.warning(
                    f"[observe_llm_execution] Failed to initialize client: {format_langfuse_error(e)}"
                )
                return func(completion_config, query, **kwargs)

            failed = False

            def langfuse_call(fn, *args, **kwargs):
                nonlocal failed
                if failed:
                    return None
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"[observe_llm_execution] {getattr(fn, '__name__', 'operation')} failed: {format_langfuse_error(e)}"
                    )
                    failed = True
                    return None

            params = completion_config.params
            trace_input = extract_query_input(query.input)
            trace = langfuse_call(
                langfuse.start_observation,
                as_type="span",
                name="unified-llm-call",
                input=trace_input,
                metadata=trace_metadata,
            )
            if trace:
                langfuse_call(
                    set_trace_attributes,
                    trace,
                    name="unified-llm-call",
                    input=trace_input,
                    session_id=session_id,
                    # OTel drops the whole list if any item is an Enum, not a plain str.
                    tags=[completion_config.provider, str(completion_config.type)],
                    metadata=trace_metadata,
                )

            generation = None
            if trace:
                generation = langfuse_call(
                    trace.start_observation,
                    as_type="generation",
                    name=f"{completion_config.provider}-completion",
                    input=_generation_input(params, trace_input),
                    model=params.get("model"),
                    model_parameters=_model_parameters(params),
                    metadata={
                        LLM_COMPLETION_TYPE: str(completion_config.type),
                        **_input_metadata(trace_input),
                    },
                )

            def finish_failure(error_msg: str) -> None:
                output = {"status": "failure", "error": error_msg}
                for observation in (generation, trace):
                    if observation:
                        langfuse_call(
                            observation.update,
                            output=output,
                            level="ERROR",
                            status_message=error_msg,
                        )
                if trace:
                    langfuse_call(
                        set_trace_attributes,
                        trace,
                        output=output,
                        session_id=session_id,
                    )
                for observation in (generation, trace):
                    if observation:
                        langfuse_call(observation.end)
                langfuse_call(langfuse.flush)

            response: LLMCallResponse | None
            error: str | None
            try:
                response, error = func(completion_config, query, **kwargs)
            except Exception as e:
                finish_failure(str(e))
                raise

            if not response:
                finish_failure(error or "Unknown error")
                return response, error

            output = {
                "status": "success",
                "output": extract_response_output(response.response.output),
            }
            if generation:
                langfuse_call(
                    generation.update,
                    output=output,
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": billable_output_tokens(response.usage),
                    },
                    model=response.response.model,
                    metadata={
                        GEN_AI_RESPONSE_ID: response.response.provider_response_id,
                        **_output_metadata(response.response.output),
                    },
                )
                langfuse_call(generation.end)
            if trace:
                langfuse_call(trace.update, output=output)
                langfuse_call(
                    set_trace_attributes,
                    trace,
                    output=output,
                    session_id=session_id or response.response.conversation_id,
                )
                langfuse_call(trace.end)

            langfuse_call(langfuse.flush)
            return response, error

        return wrapper

    return decorator
