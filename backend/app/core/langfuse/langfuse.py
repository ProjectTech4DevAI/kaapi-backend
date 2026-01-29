import uuid
import logging
from typing import Any, Callable, Dict, Optional
from functools import wraps

from asgi_correlation_id import correlation_id
from langfuse import Langfuse
from langfuse.client import StatefulGenerationClient, StatefulTraceClient
from app.models.llm import NativeCompletionConfig, QueryParams, LLMCallResponse

logger = logging.getLogger(__name__)


class LangfuseTracer:
    def __init__(
        self,
        credentials: Optional[dict] = None,
        session_id: Optional[str] = None,
        response_id: Optional[str] = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.langfuse: Optional[Langfuse] = None
        self.trace: Optional[StatefulTraceClient] = None
        self.generation: Optional[StatefulGenerationClient] = None
        self._failed = False

        has_credentials = (
            credentials
            and "public_key" in credentials
            and "secret_key" in credentials
            and "host" in credentials
        )

        if has_credentials:
            try:
                self.langfuse = Langfuse(
                    public_key=credentials["public_key"],
                    secret_key=credentials["secret_key"],
                    host=credentials["host"],
                    enabled=True,  # This ensures the client is active
                )
            except Exception as e:
                logger.warning(f"[LangfuseTracer] Failed to initialize: {e}")
                self._failed = True
                return

            if response_id:
                try:
                    traces = self.langfuse.fetch_traces(tags=response_id).data
                    if traces:
                        self.session_id = traces[0].session_id
                except Exception as e:
                    logger.debug(f"[LangfuseTracer] Session resume failed: {e}")

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
                f"[LangfuseTracer] {getattr(fn, '__name__', 'operation')} failed: {e}"
            )
            self._failed = True
            return None

    def start_trace(
        self,
        name: str,
        input: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: list[str] | None = None,
    ) -> None:
        if self._failed or not self.langfuse:
            return
        metadata = metadata or {}
        metadata["request_id"] = correlation_id.get() or "N/A"
        self.trace = self._langfuse_call(
            self.langfuse.trace,
            name=name,
            input=input,
            metadata=metadata,
            session_id=self.session_id,
            tags=tags,
        )

    def start_generation(
        self,
        name: str,
        input: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._failed or not self.trace:
            return
        self.generation = self._langfuse_call(
            self.langfuse.generation,
            name=name,
            trace_id=self.trace.id,
            input=input,
            metadata=metadata or {},
        )

    def end_generation(
        self,
        output: Dict[str, Any],
        usage: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> None:
        if self._failed or not self.generation:
            return
        self._langfuse_call(
            self.generation.end, output=output, usage=usage, model=model
        )

    def update_trace(self, tags: list[str], output: Dict[str, Any]) -> None:
        if self._failed or not self.trace:
            return
        self._langfuse_call(self.trace.update, tags=tags, output=output)

    def log_error(self, error_message: str, response_id: Optional[str] = None) -> None:
        if self._failed:
            return
        if self.generation:
            self._langfuse_call(self.generation.end, output={"error": error_message})
        if self.trace:
            self._langfuse_call(
                self.trace.update,
                tags=[response_id] if response_id else [],
                output={"status": "failure", "error": error_message},
            )

    def flush(self) -> None:
        if self._failed or not self.langfuse:
            return
        self._langfuse_call(self.langfuse.flush)


def observe_llm_execution(
    session_id: str | None = None,
    credentials: dict | None = None,
) -> Callable:
    """Decorator to add Langfuse observability to LLM provider execute methods.

    Args:
        credentials: Langfuse credentials with public_key, secret_key, and host
        session_id: Session ID for grouping traces (conversation_id)

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
            # Skip observability if no credentials provided
            if not credentials:
                logger.info(
                    "[observe_llm_execution] No credentials - skipping observability"
                )
                return func(completion_config, query, **kwargs)

            try:
                langfuse = Langfuse(
                    public_key=credentials.get("public_key"),
                    secret_key=credentials.get("secret_key"),
                    host=credentials.get("host"),
                )
                logger.info(
                    f"[observe_llm_execution] Tracing enabled | session_id={session_id or 'auto'}"
                )
            except Exception as e:
                logger.warning(
                    f"[observe_llm_execution] Failed to initialize client: {e}"
                )
                return func(completion_config, query, **kwargs)

            failed = False

            def langfuse_call(fn, *args, **kwargs):
                """Execute Langfuse operation safely. First failure disables further calls."""
                nonlocal failed
                if failed:
                    return None
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"[observe_llm_execution] {getattr(fn, '__name__', 'operation')} failed: {e}"
                    )
                    failed = True
                    return None

            trace = langfuse_call(
                langfuse.trace,
                name="unified-llm-call",
                input=query.input,
                tags=[completion_config.provider],
            )

            generation = None
            if trace:
                generation = langfuse_call(
                    trace.generation,
                    name=f"{completion_config.provider}-completion",
                    input=query.input,
                    model=completion_config.params.get("model"),
                )

            # Execute the actual LLM call
            response: LLMCallResponse | None
            error: str | None
            response, error = func(completion_config, query, **kwargs)

            if response:
                if generation:
                    langfuse_call(
                        generation.end,
                        output={
                            "status": "success",
                            "output": response.response.output.text,
                        },
                        usage_details={
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens,
                        },
                        model=response.response.model,
                    )
                if trace:
                    langfuse_call(
                        trace.update,
                        output={
                            "status": "success",
                            "output": response.response.output.text,
                        },
                        session_id=session_id or response.response.conversation_id,
                    )
            else:
                error_msg = error or "Unknown error"
                if generation:
                    langfuse_call(generation.end, output={"error": error_msg})
                if trace:
                    langfuse_call(
                        trace.update,
                        output={"status": "failure", "error": error_msg},
                        session_id=session_id,
                    )

            langfuse_call(langfuse.flush)
            return response, error

        return wrapper

    return decorator
