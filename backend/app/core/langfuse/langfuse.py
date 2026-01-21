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
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.langfuse: Langfuse = Langfuse(enabled=False)
        self.trace: Optional[StatefulTraceClient] = None
        self.generation: Optional[StatefulGenerationClient] = None

        has_credentials = (
            credentials
            and "public_key" in credentials
            and "secret_key" in credentials
            and "host" in credentials
        )
        if not has_credentials:
            logger.warning(
                "[LangfuseTracer] Missing Langfuse credentials; tracing will be disabled"
            )
            return

        try:
            self.langfuse = Langfuse(
                public_key=credentials["public_key"],
                secret_key=credentials["secret_key"],
                host=credentials["host"],
                enabled=True,  # This ensures the client is active
            )
        except Exception as e:
            self.langfuse = Langfuse(enabled=False)
            logger.error(
                f"[LangfuseTracer] Error initializing Langfuse client: {e}",
                exc_info=True,
            )
            return

        if response_id:
            try:
                traces = self.langfuse.fetch_traces(tags=response_id).data
                if traces:
                    self.session_id = traces[0].session_id

                logger.info(
                    f"[LangfuseTracer] Langfuse tracing enabled | session_id={self.session_id}"
                )
            except Exception as e:
                logger.warning(
                    f"[LangfuseTracer] Error fetching traces for response_id={response_id}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"[LangfuseTracer] Langfuse tracing initialized | session_id={self.session_id}"
        )

    def start_trace(
        self,
        name: str,
        input: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: list[str] | None = None,
    ):
        if not self.langfuse.enabled:
            return

        try:
            metadata = metadata or {}
            metadata["request_id"] = correlation_id.get() or "N/A"

            self.trace = self.langfuse.trace(
                name=name,
                input=input,
                metadata=metadata,
                session_id=self.session_id,
                tags=tags,
            )
        except Exception as e:
            logger.error(f"[LangfuseTracer] Failed to start trace: {e}")
            self.trace = None

    def start_generation(
        self,
        name: str,
        input: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.langfuse.enabled or not self.trace:
            return
        try:
            self.generation = self.langfuse.generation(
                name=name,
                trace_id=self.trace.id,
                input=input,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"[LangfuseTracer] Failed to start generation: {e}")
            self.generation = None

    def end_generation(
        self,
        output: Dict[str, Any],
        usage: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ):
        if not self.langfuse.enabled or not self.generation:
            return

        try:
            self.generation.end(output=output, usage=usage, model=model)
        except Exception as e:
            logger.error(f"[LangfuseTracer] Failed to end generation: {e}")

    def update_trace(self, tags: list[str], output: Dict[str, Any]):
        if not self.langfuse.enabled or not self.trace:
            return
        try:
            if self.trace:
                self.trace.update(tags=tags, output=output)
        except Exception as e:
            logger.error(f"[LangfuseTracer] Failed to update trace: {e}")

    def log_error(self, error_message: str, response_id: Optional[str] = None):
        if not self.langfuse.enabled:
            return
        try:
            if self.generation:
                self.generation.end(output={"error": error_message})
            if self.trace:
                self.trace.update(
                    tags=[response_id] if response_id else [],
                    output={"status": "failure", "error": error_message},
                )
        except Exception as e:
            logger.error(f"[LangfuseTracer] Failed to log error: {e}")

    def flush(self):
        if not self.langfuse.enabled:
            return
        try:
            self.langfuse.flush()
        except Exception as e:
            logger.error(f"[LangfuseTracer] Failed to flush Langfuse client: {e}")


def observe_llm_execution(
    session_id: str | None = None,
    credentials: dict | None = None,
):
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
                logger.info("[Langfuse] No credentials - skipping observability")
                return func(completion_config, query, **kwargs)

            def safe_langfuse_op(op: Callable, *args, **kwargs):
                try:
                    return op(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"[Langfuse] Operation failed: {e}")
                    return None

            try:
                langfuse = Langfuse(
                    public_key=credentials.get("public_key"),
                    secret_key=credentials.get("secret_key"),
                    host=credentials.get("host"),
                )
            except Exception as e:
                logger.warning(f"[Langfuse] Failed to initialize client: {e}")
                return func(completion_config, query, **kwargs)

            trace = safe_langfuse_op(
                langfuse.trace,
                name="unified-llm-call",
                input=query.input,
                tags=[completion_config.provider],
            )

            generation = None

            if trace:
                generation = safe_langfuse_op(
                    trace.generation,
                    name=f"{completion_config.provider}-completion",
                    input=query.input,
                    model=completion_config.params.get("model"),
                )

            try:
                # Execute the actual LLM call
                response: LLMCallResponse | None
                error: str | None
                response, error = func(completion_config, query, **kwargs)

                if response:
                    if generation:
                        safe_langfuse_op(
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
                        safe_langfuse_op(
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
                        safe_langfuse_op(generation.end, output={"error": error_msg})
                    if trace:
                        safe_langfuse_op(
                            trace.update,
                            output={"status": "failure", "error": error_msg},
                            session_id=session_id,
                        )

                safe_langfuse_op(langfuse.flush)
                return response, error

            except Exception as e:
                error_msg = str(e)
                if generation:
                    safe_langfuse_op(generation.end, output={"error": error_msg})
                if trace:
                    safe_langfuse_op(
                        trace.update,
                        output={"status": "failure", "error": error_msg},
                        session_id=session_id,
                    )
                safe_langfuse_op(langfuse.flush)
                raise

        return wrapper

    return decorator
