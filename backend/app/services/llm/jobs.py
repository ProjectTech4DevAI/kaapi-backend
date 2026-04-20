import logging
import time
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from opentelemetry import trace
from sqlmodel import Session

from app.celery.utils import start_llm_chain_job, start_llm_job
from app.core.db import engine
from app.core.langfuse.langfuse import observe_llm_execution
from app.core.telemetry import (
    set_gen_ai_request_attributes,
    set_gen_ai_response_attributes,
    flush_telemetry,
    log_context,
    record_llm_call_finished,
    record_llm_call_started,
    suppress_http_instrumentation,
)
from app.crud.config import ConfigVersionCrud
from app.crud.credentials import get_provider_credential
from app.crud.jobs import JobCrud
from app.crud.llm import create_llm_call, serialize_input, update_llm_call_response
from app.crud.llm_chain import create_llm_chain, update_llm_chain_status
from app.models import JobStatus, JobType, JobUpdate, LLMCallRequest, LLMChainRequest
from app.models.llm.request import (
    AudioInput,
    ChainStatus,
    ConfigBlob,
    ImageInput,
    KaapiCompletionConfig,
    LLMCallConfig,
    PDFInput,
    QueryParams,
    TextInput,
)
from app.models.llm.response import TextOutput
from app.services.llm.chain.types import BlockResult
from app.services.llm.guardrails import (
    list_validators_config,
    run_guardrails_validation,
)
from app.services.llm.mappers import transform_kaapi_config_to_native
from app.services.llm.providers.registry import get_llm_provider
from app.utils import APIResponse, cleanup_temp_file, resolve_input, send_callback

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def _execute_provider_call(
    *,
    func,
    completion_config: Any,
    query: QueryParams,
    credentials: dict | None,
    session_id: str | None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    kwargs.pop("organization_id", None)
    kwargs.pop("project_id", None)
    kwargs.pop("telemetry_span", None)

    decorated = observe_llm_execution(
        session_id=session_id,
        credentials=credentials,
    )(func)

    with suppress_http_instrumentation():
        return decorated(completion_config, query, **kwargs)


def start_job(
    db: Session, request: LLMCallRequest, project_id: int, organization_id: int
) -> UUID:
    """Create an LLM job and schedule Celery task."""
    with log_context(
        tag="llm-call",
        lifecycle="llm.call.start_job",
        project_id=project_id,
        organization_id=organization_id,
    ), tracer.start_as_current_span("llm.start_job") as span:
        span.set_attribute("kaapi.project_id", project_id)
        span.set_attribute("kaapi.organization_id", organization_id)

        trace_id = correlation_id.get() or "N/A"
        job_crud = JobCrud(session=db)
        job = job_crud.create(
            job_type=JobType.LLM_API, trace_id=trace_id, project_id=project_id
        )
        span.set_attribute("llm.job_id", str(job.id))

        logger.info(
            f"[start_job] Created job | job_id={job.id}, status={job.status}, project_id={project_id}"
        )

        try:
            task_id = start_llm_job(
                project_id=project_id,
                job_id=str(job.id),
                trace_id=trace_id,
                request_data=request.model_dump(mode="json"),
                organization_id=organization_id,
            )
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            logger.error(
                f"[start_job] Error starting Celery task: {str(e)} | job_id={job.id}, project_id={project_id}",
                exc_info=True,
            )
            job_update = JobUpdate(status=JobStatus.FAILED, error_message=str(e))
            job_crud.update(job_id=job.id, job_update=job_update)
            raise HTTPException(
                status_code=500, detail="Internal server error while executing LLM call"
            )

        span.set_attribute("celery.task_id", str(task_id))
        logger.info(
            f"[start_job] Job scheduled for LLM call | job_id={job.id}, project_id={project_id}, task_id={task_id}"
        )
        return job.id


def start_chain_job(
    db: Session, request: LLMChainRequest, project_id: int, organization_id: int
) -> UUID:
    """Create an LLM Chain job and schedule Celery task."""
    trace_id = correlation_id.get() or "N/A"
    job_crud = JobCrud(session=db)
    job = job_crud.create(
        job_type=JobType.LLM_CHAIN, trace_id=trace_id, project_id=project_id
    )

    with log_context(
        tag="llm-chain",
        lifecycle="llm.chain.start_job",
        job_id=job.id,
        project_id=project_id,
        organization_id=organization_id,
    ):
        logger.info(
            f"[start_chain_job] Created job | job_id={job.id}, status={job.status}, project_id={project_id}"
        )

        try:
            task_id = start_llm_chain_job(
                project_id=project_id,
                job_id=str(job.id),
                trace_id=trace_id,
                request_data=request.model_dump(mode="json"),
                organization_id=organization_id,
            )
        except Exception as e:
            logger.error(
                f"[start_chain_job] Error starting Celery task: {str(e)} | job_id={job.id}, project_id={project_id}",
                exc_info=True,
            )
            job_update = JobUpdate(status=JobStatus.FAILED, error_message=str(e))
            job_crud.update(job_id=job.id, job_update=job_update)
            raise HTTPException(
                status_code=500,
                detail="Internal server error while executing LLM chain job",
            )

        logger.info(
            f"[start_chain_job] Job scheduled for LLM chain job | job_id={job.id}, project_id={project_id}, task_id={task_id}"
        )
        return job.id


def handle_job_error(
    job_id: UUID,
    callback_url: str | None,
    callback_response: APIResponse,
) -> dict:
    """Handle job failure uniformly — send callback and update DB."""
    if callback_url:
        with tracer.start_as_current_span("llm.send_callback") as cb_span:
            cb_span.set_attribute("callback.url", callback_url)
            cb_span.set_attribute("callback.status", "failure")
            send_callback(
                callback_url=callback_url,
                data=callback_response.model_dump(),
            )

    with Session(engine) as session:
        JobCrud(session=session).update(
            job_id=job_id,
            job_update=JobUpdate(
                status=JobStatus.FAILED,
                error_message=callback_response.error,
            ),
        )

    return callback_response.model_dump()


@contextmanager
def resolved_input_context(
    query_input: TextInput | AudioInput | ImageInput | PDFInput | list,
):
    """Context manager for resolving and cleaning up input resources.

    Ensures temporary files (e.g., downloaded audio) are cleaned up
    even if errors occur during LLM execution.
    """
    resolved_input, error = resolve_input(query_input)

    if error:
        raise ValueError(error)

    try:
        yield resolved_input
    finally:
        # Clean up temp files for audio inputs
        if resolved_input and isinstance(query_input, AudioInput):
            cleanup_temp_file(resolved_input)


def resolve_config_blob(
    config_crud: ConfigVersionCrud, config: LLMCallConfig
) -> tuple[ConfigBlob | None, str | None]:
    """Fetch and parse stored config version into ConfigBlob.

    Returns:
        (config_blob, error_message)
        - config_blob: ConfigBlob if successful, else None
        - error_message: human-safe error string if an error occurs, else None
    """
    try:
        config_version = config_crud.exists_or_raise(version_number=config.version)
    except HTTPException as e:
        return None, f"Failed to retrieve stored configuration: {e.detail}"
    except Exception:
        logger.error(
            f"[resolve_config_blob] Unexpected error retrieving config version | "
            f"config_id={config.id}, version={config.version}",
            exc_info=True,
        )
        return None, "Unexpected error occurred while retrieving stored configuration"

    try:
        return ConfigBlob(**config_version.config_blob), None
    except (TypeError, ValueError) as e:
        return None, f"Stored configuration blob is invalid: {str(e)}"
    except Exception:
        logger.error(
            f"[resolve_config_blob] Unexpected error parsing config blob | "
            f"config_id={config.id}, version={config.version}",
            exc_info=True,
        )
        return None, "Unexpected error occurred while parsing stored configuration"


def apply_input_guardrails(
    *,
    config_blob: ConfigBlob | None,
    query: QueryParams,
    job_id: UUID,
    project_id: int,
    organization_id: int,
) -> tuple[QueryParams, str | None]:
    """Apply input guardrails from a config_blob. Shared with llm-call and llm-chain."""
    if not config_blob or not config_blob.input_guardrails:
        return query, None

    if not isinstance(query.input, TextInput):
        logger.info(
            f"[apply_input_guardrails] Skipping for non-text input. "
            f"job_id={job_id}, "
            f"input_type={getattr(query.input, 'type', type(query.input).__name__)}"
        )
        return query, None

    input_guardrails, _ = list_validators_config(
        organization_id=organization_id,
        project_id=project_id,
        input_validator_configs=config_blob.input_guardrails,
        output_validator_configs=None,
    )

    if not input_guardrails:
        return query, None

    safe = run_guardrails_validation(
        query.input.content.value,
        input_guardrails,
        job_id,
        project_id,
        organization_id,
        suppress_pass_logs=True,
    )

    logger.info(
        f"[apply_input_guardrails] Validation result | success={safe['success']}, job_id={job_id}"
    )

    if safe.get("bypassed"):
        logger.info(
            f"[apply_input_guardrails] Guardrails bypassed (service unavailable) | job_id={job_id}"
        )
        return query, None

    if safe["success"]:
        query.input.content.value = safe["data"]["safe_text"]
        return query, None

    return query, safe["error"]


def apply_output_guardrails(
    *,
    config_blob: ConfigBlob | None,
    result: BlockResult,
    job_id: UUID,
    project_id: int,
    organization_id: int,
) -> tuple[BlockResult, str | None]:
    """Apply output guardrails from a config_blob. Shared by /llm/call and /llm/chain.

    Returns (modified_result, None) on success, or (result, error_string) on failure.
    """
    if not config_blob or not config_blob.output_guardrails:
        return result, None

    if not isinstance(result.response.response.output, TextOutput):
        logger.info(
            f"[apply_output_guardrails] Skipping for non-text output. "
            f"job_id={job_id}, "
            f"output_type={getattr(result.response.response.output, 'type', type(result.response.response.output).__name__)}"
        )
        return result, None

    _, output_guardrails = list_validators_config(
        organization_id=organization_id,
        project_id=project_id,
        input_validator_configs=None,
        output_validator_configs=config_blob.output_guardrails,
    )

    if not output_guardrails:
        return result, None

    output_text = result.response.response.output.content.value
    safe = run_guardrails_validation(
        output_text,
        output_guardrails,
        job_id,
        project_id,
        organization_id,
        suppress_pass_logs=True,
    )

    logger.info(
        f"[apply_output_guardrails] Validation result | success={safe['success']}, job_id={job_id}"
    )

    if safe.get("bypassed"):
        logger.info(
            f"[apply_output_guardrails] Guardrails bypassed (service unavailable) | job_id={job_id}"
        )
        return result, None

    if safe["success"]:
        result.response.response.output.content.value = safe["data"]["safe_text"]
        return result, None

    return result, safe["error"]


def execute_llm_call(
    *,
    config: LLMCallConfig,
    query: QueryParams,
    job_id: UUID,
    project_id: int,
    organization_id: int,
    request_metadata: dict | None,
    langfuse_credentials: dict | None,
    include_provider_raw_response: bool = False,
    chain_id: UUID | None = None,
) -> BlockResult:
    """Execute a single LLM call. Shared by /llm/call and /llm/chain.

    Returns BlockResult with response + usage on success, or error on failure.
    """

    config_blob: ConfigBlob | None = None
    llm_call_id: UUID | None = None

    try:
        with Session(engine) as session:
            with tracer.start_as_current_span("llm.resolve_config") as cfg_span:
                cfg_span.set_attribute("llm.job_id", str(job_id))
                cfg_span.set_attribute("llm.config.is_stored", config.is_stored_config)
                if config.is_stored_config:
                    cfg_span.set_attribute("llm.config.id", str(config.id))
                    cfg_span.set_attribute("llm.config.version", str(config.version))
                    config_crud = ConfigVersionCrud(
                        session=session, project_id=project_id, config_id=config.id
                    )
                    config_blob, error = resolve_config_blob(config_crud, config)
                    if error:
                        cfg_span.set_status(trace.Status(trace.StatusCode.ERROR, error))
                        return BlockResult(error=error)
                else:
                    config_blob = config.blob

            if config_blob.prompt_template and isinstance(query.input, TextInput):
                template = config_blob.prompt_template.template
                interpolated = template.replace("{{input}}", query.input.content.value)
                query.input.content.value = interpolated

            with tracer.start_as_current_span("llm.guardrails.input") as guard_span:
                guard_span.set_attribute("llm.job_id", str(job_id))
                query, input_error = apply_input_guardrails(
                    config_blob=config_blob,
                    query=query,
                    job_id=job_id,
                    project_id=project_id,
                    organization_id=organization_id,
                )
                if input_error:
                    guard_span.set_status(
                        trace.Status(trace.StatusCode.ERROR, input_error)
                    )
                    return BlockResult(error=input_error)

            completion_config = config_blob.completion
            original_provider = completion_config.provider

            if isinstance(completion_config, KaapiCompletionConfig):
                completion_config, warnings = transform_kaapi_config_to_native(
                    session=session, kaapi_config=completion_config
                )
                if request_metadata is None:
                    request_metadata = {}
                request_metadata.setdefault("warnings", []).extend(warnings)

            model_name = str(completion_config.params.get("model") or "")

            resolved_config_blob = ConfigBlob(
                completion=completion_config,
                prompt_template=config_blob.prompt_template,
                input_guardrails=config_blob.input_guardrails,
                output_guardrails=config_blob.output_guardrails,
            )

            with tracer.start_as_current_span("llm.create_call_record") as create_span:
                create_span.set_attribute("llm.job_id", str(job_id))
                create_span.set_attribute(
                    "llm.provider", str(completion_config.provider)
                )
                if model_name:
                    create_span.set_attribute("llm.request.model", model_name)
                try:
                    llm_call_request = LLMCallRequest(
                        query=query,
                        config=config,
                        request_metadata=request_metadata,
                    )
                    llm_call = create_llm_call(
                        session,
                        request=llm_call_request,
                        job_id=job_id,
                        project_id=project_id,
                        organization_id=organization_id,
                        resolved_config=resolved_config_blob,
                        original_provider=original_provider,
                        chain_id=chain_id,
                    )
                    llm_call_id = llm_call.id
                    create_span.set_attribute("llm.call_id", str(llm_call_id))
                    logger.info(
                        f"[execute_llm_call] Created LLM call record | "
                        f"llm_call_id={llm_call_id}, job_id={job_id}"
                    )
                except Exception as e:
                    create_span.record_exception(e)
                    create_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    logger.error(
                        f"[execute_llm_call] Failed to create LLM call record: {e} | job_id={job_id}",
                        exc_info=True,
                    )
                    return BlockResult(
                        error=f"Failed to create LLM call record: {str(e)}"
                    )

            try:
                provider_instance = get_llm_provider(
                    session=session,
                    provider_type=completion_config.provider,
                    project_id=project_id,
                    organization_id=organization_id,
                )
            except ValueError as ve:
                return BlockResult(error=str(ve), llm_call_id=llm_call_id)

        conversation_id = None
        if query.conversation and query.conversation.id:
            conversation_id = query.conversation.id

        operation = "chat"
        provider_name = str(completion_config.provider)
        model_name = str(completion_config.params.get("model") or "")
        completion_type = str(completion_config.type or "")
        record_llm_call_started(
            provider=provider_name,
            model=model_name,
            operation=operation,
            organization_id=organization_id,
            project_id=project_id,
        )
        provider_started_at = time.perf_counter()
        response = None
        error = None

        # Wrap the provider call in a `chat <model>` span so Sentry's AI Insights
        # module recognises it (op=gen_ai.chat) and surfaces tokens / model /
        # messages on the trace. This is the span AI Insights keys off — keep it
        # as the parent of `llm.provider.execute`.
        ai_span_name = f"chat {model_name}" if model_name else f"chat {provider_name}"
        with tracer.start_as_current_span(ai_span_name) as ai_span:
            ai_span.set_attribute("sentry.op", "gen_ai.chat")
            ai_span.set_attribute("gen_ai.request.organization_id", organization_id)
            ai_span.set_attribute("gen_ai.request.project_id", project_id)
            if completion_type:
                ai_span.set_attribute("completion_type", completion_type)
            set_gen_ai_request_attributes(
                ai_span,
                provider=provider_name,
                model=model_name,
                operation=operation,
                organization_id=organization_id,
                project_id=project_id,
                params=completion_config.params or {},
            )

            try:
                with resolved_input_context(query.input) as resolved_input:
                    with tracer.start_as_current_span(
                        "llm.provider.execute"
                    ) as provider_span:
                        provider_span.set_attribute("llm.provider", provider_name)
                        provider_span.set_attribute(
                            "llm.operation.name", "provider.execute"
                        )
                        if completion_type:
                            provider_span.set_attribute(
                                "completion_type", completion_type
                            )
                        if model_name:
                            provider_span.set_attribute("llm.request.model", model_name)

                        response, error = _execute_provider_call(
                            func=provider_instance.execute,
                            completion_config=completion_config,
                            query=query,
                            credentials=langfuse_credentials,
                            session_id=conversation_id,
                            organization_id=organization_id,
                            project_id=project_id,
                            telemetry_span=provider_span,
                            resolved_input=resolved_input,
                            include_provider_raw_response=include_provider_raw_response,
                        )
            except ValueError as ve:
                ai_span.set_status(trace.Status(trace.StatusCode.ERROR, str(ve)))
                record_llm_call_finished(
                    provider=provider_name,
                    model=model_name,
                    operation=operation,
                    duration_ms=(time.perf_counter() - provider_started_at) * 1000,
                    error=True,
                    organization_id=organization_id,
                    project_id=project_id,
                )
                return BlockResult(error=str(ve), llm_call_id=llm_call_id)

            if response:
                set_gen_ai_response_attributes(ai_span, response=response)
                usage = response.usage
                if usage:
                    ai_span.set_attribute("llm.usage.total_tokens", usage.total_tokens)
                    ai_span.set_attribute("kaapi_llm_input_tokens", usage.input_tokens)
                    ai_span.set_attribute(
                        "kaapi_llm_output_tokens", usage.output_tokens
                    )
                    ai_span.set_attribute("kaapi_llm_total_tokens", usage.total_tokens)
            else:
                ai_span.set_status(
                    trace.Status(trace.StatusCode.ERROR, error or "Unknown error")
                )

        # Push the just-finished LLM span promptly instead of waiting for task teardown.
        flush_telemetry(timeout_millis=10000)

        if response:
            with Session(engine) as session:
                if llm_call_id:
                    with tracer.start_as_current_span(
                        "llm.update_call_record"
                    ) as update_span:
                        update_span.set_attribute("llm.call_id", str(llm_call_id))
                        update_span.set_attribute("llm.job_id", str(job_id))
                        try:
                            update_llm_call_response(
                                session,
                                llm_call_id=llm_call_id,
                                provider_response_id=response.response.provider_response_id,
                                content=response.response.output.model_dump(),
                                usage=response.usage.model_dump(),
                                conversation_id=response.response.conversation_id,
                            )
                        except Exception as e:
                            update_span.record_exception(e)
                            update_span.set_status(
                                trace.Status(trace.StatusCode.ERROR, str(e))
                            )
                            logger.error(
                                f"[execute_llm_call] Failed to update LLM call record: {e} | "
                                f"llm_call_id={llm_call_id}",
                                exc_info=True,
                            )

            duration_ms = (time.perf_counter() - provider_started_at) * 1000
            record_llm_call_finished(
                provider=provider_name,
                model=model_name,
                operation=operation,
                duration_ms=duration_ms,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                error=False,
                organization_id=organization_id,
                project_id=project_id,
            )

            result = BlockResult(
                response=response,
                llm_call_id=llm_call_id,
                usage=response.usage,
                metadata=request_metadata,
            )

            with tracer.start_as_current_span(
                "llm.guardrails.output"
            ) as out_guard_span:
                out_guard_span.set_attribute("llm.job_id", str(job_id))
                result, output_error = apply_output_guardrails(
                    config_blob=config_blob,
                    result=result,
                    job_id=job_id,
                    project_id=project_id,
                    organization_id=organization_id,
                )
                if output_error:
                    out_guard_span.set_status(
                        trace.Status(trace.StatusCode.ERROR, output_error)
                    )
                    return BlockResult(error=output_error, llm_call_id=llm_call_id)

            return result

        duration_ms = (time.perf_counter() - provider_started_at) * 1000
        record_llm_call_finished(
            provider=provider_name,
            model=model_name,
            operation=operation,
            duration_ms=duration_ms,
            error=True,
            organization_id=organization_id,
            project_id=project_id,
        )
        error_message = error or "Unknown error occurred"
        return BlockResult(error=error_message, llm_call_id=llm_call_id)

    except Exception as e:
        logger.error(
            f"[execute_llm_call] Unexpected error: {e} | job_id={job_id}",
            exc_info=True,
        )
        return BlockResult(
            error="Unexpected error occurred",
            llm_call_id=llm_call_id,
        )


def execute_job(
    request_data: dict,
    project_id: int,
    organization_id: int,
    job_id: str,
    task_id: str,
    task_instance,
) -> dict:
    """Celery task to process an LLM request asynchronously.

    Returns:
        dict: Serialized APIResponse[LLMCallResponse] on success, APIResponse[None] on failure
    """
    request = LLMCallRequest(**request_data)
    job_uuid = UUID(job_id)
    callback_url_str = str(request.callback_url) if request.callback_url else None

    with log_context(
        tag="llm-call",
        lifecycle="llm.call.execute_job",
        job_id=job_uuid,
        task_id=task_id,
        project_id=project_id,
        organization_id=organization_id,
    ):
        logger.info(
            f"[execute_job] Starting LLM job execution | job_id={job_id}, task_id={task_id}, callback_url {callback_url_str}"
        )

        try:
            with Session(engine) as session:
                job_crud = JobCrud(session=session)
                job_crud.update(
                    job_id=job_uuid, job_update=JobUpdate(status=JobStatus.PROCESSING)
                )

                langfuse_credentials = get_provider_credential(
                    session=session,
                    org_id=organization_id,
                    project_id=project_id,
                    provider="langfuse",
                )

            result = execute_llm_call(
                config=request.config,
                query=request.query,
                job_id=job_uuid,
                project_id=project_id,
                organization_id=organization_id,
                request_metadata=request.request_metadata,
                langfuse_credentials=langfuse_credentials,
                include_provider_raw_response=request.include_provider_raw_response,
            )

            logger.info(
                f"[execute_job] Error if any during execution of job: {result.error}"
            )

            if result.success:
                callback_response = APIResponse.success_response(
                    data=result.response, metadata=result.metadata
                )
                if callback_url_str:
                    with tracer.start_as_current_span("llm.send_callback") as cb_span:
                        cb_span.set_attribute("callback.url", callback_url_str)
                        cb_span.set_attribute("callback.status", "success")
                        cb_span.set_attribute("llm.job_id", str(job_uuid))
                        send_callback(
                            callback_url=callback_url_str,
                            data=callback_response.model_dump(),
                        )

                with Session(engine) as session:
                    JobCrud(session=session).update(
                        job_id=job_uuid, job_update=JobUpdate(status=JobStatus.SUCCESS)
                    )
                    logger.info(
                        f"[execute_job] Successfully completed LLM job | job_id={job_id}, "
                        f"tokens={result.usage.total_tokens}"
                    )
                    return callback_response.model_dump()

            error_message = result.error or "Unknown error occurred"
            callback_response = APIResponse.failure_response(
                error=error_message,
                metadata=request.request_metadata,
            )
            return handle_job_error(job_uuid, callback_url_str, callback_response)

        except Exception as e:
            callback_response = APIResponse.failure_response(
                error="Unexpected error occurred",
                metadata=request.request_metadata,
            )
            logger.error(
                f"[execute_job] Unexpected error: {str(e)} | job_id={job_uuid}, task_id={task_id}",
                exc_info=True,
            )
            return handle_job_error(job_uuid, callback_url_str, callback_response)
        finally:
            # Ensure task spans are pushed promptly so Sentry dashboards update faster.
            flush_telemetry()


def execute_chain_job(
    request_data: dict,
    project_id: int,
    organization_id: int,
    job_id: str,
    task_id: str,
    task_instance,
) -> dict:
    """Celery task to process an LLM Chain request asynchronously.

    Returns:
        dict: Serialized APIResponse[LLMChainResponse] on success, APIResponse[None] on failure
    """
    # imports to avoid circular dependency:
    from app.services.llm.chain.chain import ChainBlock, ChainContext, LLMChain
    from app.services.llm.chain.executor import ChainExecutor

    request = LLMChainRequest(**request_data)
    job_uuid = UUID(job_id)
    callback_url_str = str(request.callback_url) if request.callback_url else None
    chain_uuid = None

    with log_context(
        tag="llm-chain",
        lifecycle="llm.chain.execute_job",
        job_id=job_uuid,
        task_id=task_id,
        project_id=project_id,
        organization_id=organization_id,
        total_blocks=len(request.blocks),
    ):
        logger.info(
            f"[execute_chain_job] Starting chain execution | "
            f"job_id={job_uuid}, total_blocks={len(request.blocks)}"
        )

        try:
            with Session(engine) as session:
                chain_record = create_llm_chain(
                    session,
                    job_id=job_uuid,
                    project_id=project_id,
                    organization_id=organization_id,
                    total_blocks=len(request.blocks),
                    input=serialize_input(request.query.input),
                    configs=[block.model_dump(mode="json") for block in request.blocks],
                )
                chain_uuid = chain_record.id

                logger.info(
                    f"[execute_chain_job] Created chain record | "
                    f"chain_id={chain_uuid}, job_id={job_uuid}"
                )

                langfuse_credentials = get_provider_credential(
                    session=session,
                    org_id=organization_id,
                    project_id=project_id,
                    provider="langfuse",
                )

            context = ChainContext(
                job_id=job_uuid,
                chain_id=chain_uuid,
                project_id=project_id,
                organization_id=organization_id,
                langfuse_credentials=langfuse_credentials,
                request_metadata=request.request_metadata,
                total_blocks=len(request.blocks),
                callback_url=str(request.callback_url)
                if request.callback_url
                else None,
                intermediate_callback_flags=[
                    block.intermediate_callback for block in request.blocks
                ],
            )

            blocks = [
                ChainBlock(
                    config=block.config,
                    index=i,
                    context=context,
                    include_provider_raw_response=block.include_provider_raw_response,
                )
                for i, block in enumerate(request.blocks)
            ]

            chain = LLMChain(blocks, context)

            executor = ChainExecutor(chain=chain, context=context, request=request)
            return executor.run()

        except Exception as e:
            logger.error(
                f"[execute_chain_job] Failed: {e} | job_id={job_uuid}",
                exc_info=True,
            )

            if chain_uuid:
                try:
                    with Session(engine) as session:
                        update_llm_chain_status(
                            session,
                            chain_id=chain_uuid,
                            status=ChainStatus.FAILED,
                            error=str(e),
                        )
                except Exception:
                    logger.error(
                        f"[execute_chain_job] Failed to update chain status: {e} | "
                        f"chain_id={chain_uuid}",
                        exc_info=True,
                    )

            callback_response = APIResponse.failure_response(
                error="Unexpected error occurred",
                metadata=request.request_metadata,
            )
            return handle_job_error(job_uuid, callback_url_str, callback_response)
        finally:
            # Ensure task spans are pushed promptly so Sentry dashboards update faster.
            flush_telemetry()
