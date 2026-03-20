import logging
import time
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from sqlmodel import Session

from app.celery.utils import start_high_priority_job
from app.core.db import engine
from app.core.langfuse.langfuse import observe_llm_execution
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


def start_job(
    db: Session,
    request: LLMCallRequest,
    project_id: int,
    organization_id: int,
    api_start_time_wall: float | None = None,
) -> UUID:
    """Create an LLM job and schedule Celery task."""
    trace_id = correlation_id.get() or "N/A"

    t_create_start = time.perf_counter()
    job_crud = JobCrud(session=db)
    job = job_crud.create(job_type=JobType.LLM_API, trace_id=trace_id)

    # Explicitly flush to ensure job is persisted before Celery task starts
    db.flush()
    db.commit()
    t_create = (time.perf_counter() - t_create_start) * 1000

    logger.info(
        f"[start_job] Created job | job_id={job.id}, status={job.status}, project_id={project_id}"
    )

    try:
        t_dispatch_start = time.perf_counter()
        task_id = start_high_priority_job(
            function_path="app.services.llm.jobs.execute_job",
            project_id=project_id,
            job_id=str(job.id),
            trace_id=trace_id,
            request_data=request.model_dump(mode="json"),
            organization_id=organization_id,
            api_start_time_wall=api_start_time_wall,  # Wall-clock time for cross-process timing
        )
        t_dispatch = (time.perf_counter() - t_dispatch_start) * 1000

        if api_start_time_wall:
            # Use wall-clock time for accurate cross-process measurement
            time_since_api_start = (time.time() - api_start_time_wall) * 1000
            logger.info(
                f"[E2E_TIMING] Job dispatched to Celery | "
                f"job_create_time={t_create:.2f}ms, "
                f"dispatch_time={t_dispatch:.2f}ms, "
                f"total_from_api_start={time_since_api_start:.2f}ms | "
                f"job_id={job.id}"
            )
    except Exception as e:
        logger.error(
            f"[start_job] Error starting Celery task: {str(e)} | job_id={job.id}, project_id={project_id}",
            exc_info=True,
        )
        job_update = JobUpdate(status=JobStatus.FAILED, error_message=str(e))
        job_crud.update(job_id=job.id, job_update=job_update)
        raise HTTPException(
            status_code=500, detail="Internal server error while executing LLM call"
        )

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
    job = job_crud.create(job_type=JobType.LLM_CHAIN, trace_id=trace_id)

    # Explicitly flush to ensure job is persisted before Celery task starts
    db.flush()
    db.commit()

    logger.info(
        f"[start_chain_job] Created job | job_id={job.id}, status={job.status}, project_id={project_id}"
    )

    try:
        task_id = start_high_priority_job(
            function_path="app.services.llm.jobs.execute_chain_job",
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
    with Session(engine) as session:
        job_crud = JobCrud(session=session)

        if callback_url:
            send_callback(
                callback_url=callback_url,
                data=callback_response.model_dump(),
            )

        job_crud.update(
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
    call_start = time.perf_counter()
    timings = {}

    config_blob: ConfigBlob | None = None
    llm_call_id: UUID | None = None

    try:
        t_start = time.perf_counter()
        with Session(engine) as session:
            t_session_create = time.perf_counter()
            timings["db_session_config_create"] = (t_session_create - t_start) * 1000
            t_config_start = time.perf_counter()
            if config.is_stored_config:
                config_crud = ConfigVersionCrud(
                    session=session, project_id=project_id, config_id=config.id
                )
                config_blob, error = resolve_config_blob(config_crud, config)
                if error:
                    return BlockResult(error=error)
            else:
                config_blob = config.blob
            timings["config_resolution"] = (time.perf_counter() - t_config_start) * 1000

            t_template_start = time.perf_counter()
            if config_blob.prompt_template and isinstance(query.input, TextInput):
                template = config_blob.prompt_template.template
                interpolated = template.replace("{{input}}", query.input.content.value)
                query.input.content.value = interpolated
            timings["prompt_template_interpolation"] = (
                time.perf_counter() - t_template_start
            ) * 1000

            t_guardrail_start = time.perf_counter()
            query, input_error = apply_input_guardrails(
                config_blob=config_blob,
                query=query,
                job_id=job_id,
                project_id=project_id,
                organization_id=organization_id,
            )
            timings["input_guardrails"] = (
                time.perf_counter() - t_guardrail_start
            ) * 1000
            if input_error:
                return BlockResult(error=input_error)

            t_config_transform_start = time.perf_counter()
            completion_config = config_blob.completion
            original_provider = completion_config.provider

            if isinstance(completion_config, KaapiCompletionConfig):
                completion_config, warnings = transform_kaapi_config_to_native(
                    completion_config
                )
                if request_metadata is None:
                    request_metadata = {}
                request_metadata.setdefault("warnings", []).extend(warnings)

            resolved_config_blob = ConfigBlob(
                completion=completion_config,
                prompt_template=config_blob.prompt_template,
                input_guardrails=config_blob.input_guardrails,
                output_guardrails=config_blob.output_guardrails,
            )
            timings["config_transformation"] = (
                time.perf_counter() - t_config_transform_start
            ) * 1000

            try:
                t_llm_call_create_start = time.perf_counter()
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
                timings["create_llm_call_record"] = (
                    time.perf_counter() - t_llm_call_create_start
                ) * 1000
                logger.info(
                    f"[execute_llm_call] Created LLM call record | "
                    f"llm_call_id={llm_call_id}, job_id={job_id}"
                )
            except Exception as e:
                logger.error(
                    f"[execute_llm_call] Failed to create LLM call record: {e} | job_id={job_id}",
                    exc_info=True,
                )
                return BlockResult(error=f"Failed to create LLM call record: {str(e)}")

            try:
                t_provider_start = time.perf_counter()
                provider_instance = get_llm_provider(
                    session=session,
                    provider_type=completion_config.provider,
                    project_id=project_id,
                    organization_id=organization_id,
                )
                timings["get_provider_instance"] = (
                    time.perf_counter() - t_provider_start
                ) * 1000
            except ValueError as ve:
                return BlockResult(error=str(ve), llm_call_id=llm_call_id)

        timings["db_session_config_total"] = (time.perf_counter() - t_start) * 1000

        t_conversation_start = time.perf_counter()
        conversation_id = None
        if query.conversation and query.conversation.id:
            conversation_id = query.conversation.id
        timings["conversation_id_extract"] = (
            time.perf_counter() - t_conversation_start
        ) * 1000

        # Apply Langfuse observability decorator to provider execute method
        t_decorator_start = time.perf_counter()
        decorated_execute = observe_llm_execution(
            credentials=langfuse_credentials,
            session_id=conversation_id,
        )(provider_instance.execute)
        timings["langfuse_decorator_apply"] = (
            time.perf_counter() - t_decorator_start
        ) * 1000

        # Resolve input and execute LLM (context manager handles cleanup)
        try:
            t_api_call_start = time.perf_counter()
            with resolved_input_context(query.input) as resolved_input:
                t_input_resolved = time.perf_counter()
                timings["input_resolution"] = (
                    t_input_resolved - t_api_call_start
                ) * 1000

                response, error = decorated_execute(
                    completion_config=completion_config,
                    query=query,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
                t_api_call_end = time.perf_counter()
                timings["actual_llm_api_call"] = (
                    t_api_call_end - t_input_resolved
                ) * 1000

            timings["llm_execution_total"] = (
                time.perf_counter() - t_api_call_start
            ) * 1000
            logger.info(
                f"[TIMING] Actual LLM API call | duration={timings['actual_llm_api_call']:.2f}ms | job_id={job_id}"
            )
        except ValueError as ve:
            return BlockResult(error=str(ve), llm_call_id=llm_call_id)

        if response:
            t_update_start = time.perf_counter()
            with Session(engine) as session:
                t_session_create = time.perf_counter()
                timings["db_session_response_create"] = (
                    t_session_create - t_update_start
                ) * 1000

                if llm_call_id:
                    try:
                        t_update_call_start = time.perf_counter()
                        update_llm_call_response(
                            session,
                            llm_call_id=llm_call_id,
                            provider_response_id=response.response.provider_response_id,
                            content=response.response.output.model_dump(),
                            usage=response.usage.model_dump(),
                            conversation_id=response.response.conversation_id,
                        )
                        timings["update_llm_call_response_db"] = (
                            time.perf_counter() - t_update_call_start
                        ) * 1000
                    except Exception as e:
                        logger.error(
                            f"[execute_llm_call] Failed to update LLM call record: {e} | "
                            f"llm_call_id={llm_call_id}",
                            exc_info=True,
                        )
            timings["db_session_response_total"] = (
                time.perf_counter() - t_update_start
            ) * 1000

            t_result_build_start = time.perf_counter()
            result = BlockResult(
                response=response,
                llm_call_id=llm_call_id,
                usage=response.usage,
                metadata=request_metadata,
            )
            timings["build_block_result"] = (
                time.perf_counter() - t_result_build_start
            ) * 1000

            t_output_guardrail_start = time.perf_counter()
            result, output_error = apply_output_guardrails(
                config_blob=config_blob,
                result=result,
                job_id=job_id,
                project_id=project_id,
                organization_id=organization_id,
            )
            timings["output_guardrails"] = (
                time.perf_counter() - t_output_guardrail_start
            ) * 1000
            if output_error:
                return BlockResult(error=output_error, llm_call_id=llm_call_id)

            total_time = (time.perf_counter() - call_start) * 1000

            # Helper function for red highlighting if >1000ms
            def format_time(ms: float) -> str:
                if ms > 1000:
                    return f"\033[91m{ms:>8.2f}ms ⚠️\033[0m"  # Red color
                return f"{ms:>8.2f}ms"

            # Log timing summary
            logger.info(
                f"[TIMING] ═══ EXECUTE_LLM_CALL TIMING SUMMARY (job_id={job_id}) ═══"
            )
            logger.info(
                f"[TIMING]   DB Session (config):           {format_time(timings['db_session_config_total'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Session create:           {format_time(timings['db_session_config_create'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Config resolution:        {format_time(timings['config_resolution'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Template interpolation:   {format_time(timings['prompt_template_interpolation'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Input guardrails:         {format_time(timings['input_guardrails'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Config transformation:    {format_time(timings['config_transformation'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Create LLM call record:   {format_time(timings['create_llm_call_record'])}"
            )
            logger.info(
                f"[TIMING]     └─ Get provider instance:    {format_time(timings['get_provider_instance'])}"
            )
            logger.info(
                f"[TIMING]   Langfuse decorator:            {format_time(timings['langfuse_decorator_apply'])}"
            )
            logger.info(
                f"[TIMING]   LLM Execution:                 {format_time(timings['llm_execution_total'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Input resolution:         {format_time(timings['input_resolution'])}"
            )
            logger.info(
                f"[TIMING]     └─ ACTUAL LLM API CALL:      {format_time(timings['actual_llm_api_call'])} ⚡"
            )
            logger.info(
                f"[TIMING]   DB Session (response):         {format_time(timings['db_session_response_total'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Session create:           {format_time(timings['db_session_response_create'])}"
            )
            logger.info(
                f"[TIMING]     └─ Update LLM call response: {format_time(timings['update_llm_call_response_db'])}"
            )
            logger.info(
                f"[TIMING]   Output guardrails:             {format_time(timings['output_guardrails'])}"
            )
            logger.info(
                f"[TIMING] ════════════════════════════════════════════════════"
            )
            logger.info(
                f"[TIMING]   TOTAL execute_llm_call:        {format_time(total_time)}"
            )
            logger.info(
                f"[TIMING] ════════════════════════════════════════════════════"
            )

            return result

        return BlockResult(
            error=error or "Unknown error occurred",
            llm_call_id=llm_call_id,
        )

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
    api_start_time_wall: float | None = None,
) -> dict:
    """Celery task to process an LLM request asynchronously.

    Returns:
        dict: Serialized APIResponse[LLMCallResponse] on success, APIResponse[None] on failure
    """
    # Use perf_counter for local worker timing (high precision)
    job_start_local = time.perf_counter()
    # Capture wall-clock time for cross-process comparison
    job_start_wall = time.time()
    timings = {}

    # Calculate queue wait time (time from API dispatch to worker pickup)
    if api_start_time_wall:
        # Use wall-clock time for accurate cross-process measurement
        queue_wait_time = (job_start_wall - api_start_time_wall) * 1000
        logger.info(
            f"[E2E_TIMING] ═══ CELERY WORKER STARTED PROCESSING ═══ | "
            f"queue_wait_time={queue_wait_time:.2f}ms | "
            f"job_id={job_id}"
        )
        timings["queue_wait_time"] = queue_wait_time

    t_start = time.perf_counter()
    request = LLMCallRequest(**request_data)
    job_uuid = UUID(job_id)  # Renamed to avoid shadowing parameter
    callback_url_str = str(request.callback_url) if request.callback_url else None
    timings["request_parsing"] = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"[execute_job] Starting LLM job execution | job_id={job_id}, task_id={task_id}, callback_url {callback_url_str}"
    )

    try:
        t_start = time.perf_counter()
        with Session(engine) as session:
            t_session_create = time.perf_counter()
            timings["db_session_1_create"] = (t_session_create - t_start) * 1000

            job_crud = JobCrud(session=session)
            t_crud_init = time.perf_counter()
            timings["job_crud_init"] = (t_crud_init - t_session_create) * 1000

            job_crud.update(
                job_id=job_uuid, job_update=JobUpdate(status=JobStatus.PROCESSING)
            )
            t_status_update = time.perf_counter()
            timings["job_status_update_processing"] = (
                t_status_update - t_crud_init
            ) * 1000

            langfuse_credentials = get_provider_credential(
                session=session,
                org_id=organization_id,
                project_id=project_id,
                provider="langfuse",
            )
            t_langfuse_fetch = time.perf_counter()
            timings["langfuse_credentials_fetch"] = (
                t_langfuse_fetch - t_status_update
            ) * 1000

        timings["db_session_1_total"] = (time.perf_counter() - t_start) * 1000

        logger.info(
            f"[TIMING] DB Session 1 (status update + langfuse) | duration={timings['db_session_1_total']:.2f}ms | job_id={job_id}"
        )

        t_start = time.perf_counter()
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
        timings["execute_llm_call"] = (time.perf_counter() - t_start) * 1000

        logger.info(
            f"[TIMING] execute_llm_call | duration={timings['execute_llm_call']:.2f}ms | job_id={job_id}"
        )
        logger.info(
            f"[execute_job] Error if any during execution of job: {result.error}"
        )

        if result.success:
            t_start = time.perf_counter()
            callback_response = APIResponse.success_response(
                data=result.response, metadata=result.metadata
            )
            timings["callback_response_build"] = (time.perf_counter() - t_start) * 1000

            if callback_url_str:
                t_start = time.perf_counter()
                send_callback(
                    callback_url=callback_url_str,
                    data=callback_response.model_dump(),
                )
                timings["callback_delivery"] = (time.perf_counter() - t_start) * 1000

                # Track when callback was sent (for E2E timing)
                if api_start_time_wall:
                    timings["time_to_callback_sent"] = (
                        time.time() - api_start_time_wall
                    ) * 1000

                logger.info(
                    f"[TIMING] Callback delivery | duration={timings['callback_delivery']:.2f}ms | job_id={job_id}"
                )

            t_start = time.perf_counter()
            with Session(engine) as session:
                t_session_create = time.perf_counter()
                timings["db_session_2_create"] = (t_session_create - t_start) * 1000

                JobCrud(session=session).update(
                    job_id=job_uuid, job_update=JobUpdate(status=JobStatus.SUCCESS)
                )
                t_status_update = time.perf_counter()
                timings["job_status_update_success"] = (
                    t_status_update - t_session_create
                ) * 1000

            timings["db_session_2_total"] = (time.perf_counter() - t_start) * 1000

            # Calculate worker execution time (local to this process)
            total_time = (time.perf_counter() - job_start_local) * 1000

            # Calculate end-to-end time using wall-clock time (cross-process)
            if api_start_time_wall:
                end_to_end_time = (time.time() - api_start_time_wall) * 1000
                timings["end_to_end_total"] = end_to_end_time

            # Helper function for red highlighting if >1000ms
            def format_time(ms: float) -> str:
                if ms > 1000:
                    return f"\033[91m{ms:>8.2f}ms ⚠️\033[0m"  # Red color
                return f"{ms:>8.2f}ms"

            # Log timing summary
            logger.info(
                f"[TIMING] ═══ EXECUTE_JOB TIMING SUMMARY (job_id={job_id}) ═══"
            )
            if "queue_wait_time" in timings:
                logger.info(
                    f"[TIMING]   Queue wait time:              {format_time(timings['queue_wait_time'])} ⏱️"
                )
            logger.info(
                f"[TIMING]   Request parsing:              {format_time(timings['request_parsing'])}"
            )
            logger.info(
                f"[TIMING]   DB Session 1 (total):         {format_time(timings['db_session_1_total'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Session create:          {format_time(timings['db_session_1_create'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Status → PROCESSING:     {format_time(timings['job_status_update_processing'])}"
            )
            logger.info(
                f"[TIMING]     └─ Langfuse creds fetch:    {format_time(timings['langfuse_credentials_fetch'])}"
            )
            logger.info(
                f"[TIMING]   Execute LLM call:             {format_time(timings['execute_llm_call'])}"
            )
            if "callback_delivery" in timings:
                logger.info(
                    f"[TIMING]   Callback delivery:            {format_time(timings['callback_delivery'])}"
                )
            logger.info(
                f"[TIMING]   DB Session 2 (total):         {format_time(timings['db_session_2_total'])}"
            )
            logger.info(
                f"[TIMING]     ├─ Session create:          {format_time(timings['db_session_2_create'])}"
            )
            logger.info(
                f"[TIMING]     └─ Status → SUCCESS:        {format_time(timings['job_status_update_success'])}"
            )
            logger.info(
                f"[TIMING] ════════════════════════════════════════════════════"
            )
            logger.info(
                f"[TIMING]   TOTAL execute_job:            {format_time(total_time)}"
            )
            logger.info(
                f"[TIMING] ════════════════════════════════════════════════════"
            )

            # Log end-to-end timing
            if "end_to_end_total" in timings:
                logger.info(
                    f"[E2E_TIMING] ═══════════════════════════════════════════════════════════"
                )
                if "time_to_callback_sent" in timings:
                    logger.info(
                        f"[E2E_TIMING] 🎯 TIME TO CALLBACK SENT (API → Callback): {format_time(timings['time_to_callback_sent'])}"
                    )
                logger.info(
                    f"[E2E_TIMING] 🎯 TOTAL END-TO-END TIME (API → Job Complete): {format_time(timings['end_to_end_total'])}"
                )
                logger.info(
                    f"[E2E_TIMING]    ├─ Queue wait time:           {format_time(timings['queue_wait_time'])} ({timings['queue_wait_time']/timings['end_to_end_total']*100:.1f}%)"
                )
                logger.info(
                    f"[E2E_TIMING]    └─ Worker execution:          {format_time(total_time)} ({total_time/timings['end_to_end_total']*100:.1f}%)"
                )

                # Show accurate worker execution breakdown
                logger.info(f"[E2E_TIMING]")
                logger.info(f"[E2E_TIMING] Worker Execution Breakdown:")

                # Calculate overhead (everything except execute_llm_call and callback)
                execute_llm_call_time = timings.get("execute_llm_call", 0)
                callback_time = timings.get("callback_delivery", 0)
                db_session_1_time = timings.get("db_session_1_total", 0)
                db_session_2_time = timings.get("db_session_2_total", 0)
                request_parsing_time = timings.get("request_parsing", 0)

                worker_overhead = (
                    request_parsing_time + db_session_1_time + db_session_2_time
                )

                logger.info(
                    f"[E2E_TIMING]    ├─ Request parsing + DB ops:  {format_time(worker_overhead)}"
                )
                logger.info(
                    f"[E2E_TIMING]    │   ├─ Request parsing:        {format_time(request_parsing_time)}"
                )
                logger.info(
                    f"[E2E_TIMING]    │   ├─ DB Session 1:           {format_time(db_session_1_time)}"
                )
                logger.info(
                    f"[E2E_TIMING]    │   └─ DB Session 2:           {format_time(db_session_2_time)}"
                )
                logger.info(
                    f"[E2E_TIMING]    ├─ execute_llm_call (total):  {format_time(execute_llm_call_time)}"
                )
                logger.info(f"[E2E_TIMING]    │   └─ (see detailed breakdown above)")
                logger.info(
                    f"[E2E_TIMING]    └─ Callback delivery:         {format_time(callback_time)}"
                )

                # Show percentages
                logger.info(f"[E2E_TIMING]")
                logger.info(f"[E2E_TIMING] Time Distribution:")
                logger.info(
                    f"[E2E_TIMING]    Queue wait:        {format_time(timings['queue_wait_time'])} ({timings['queue_wait_time']/timings['end_to_end_total']*100:>5.1f}%)"
                )
                logger.info(
                    f"[E2E_TIMING]    Worker overhead:   {format_time(worker_overhead)} ({worker_overhead/timings['end_to_end_total']*100:>5.1f}%)"
                )
                logger.info(
                    f"[E2E_TIMING]    execute_llm_call:  {format_time(execute_llm_call_time)} ({execute_llm_call_time/timings['end_to_end_total']*100:>5.1f}%)"
                )
                logger.info(
                    f"[E2E_TIMING]    Callback delivery: {format_time(callback_time)} ({callback_time/timings['end_to_end_total']*100:>5.1f}%)"
                )

                logger.info(
                    f"[E2E_TIMING] ═══════════════════════════════════════════════════════════"
                )

            logger.info(
                f"[execute_job] Successfully completed LLM job | job_id={job_id}, "
                f"tokens={result.usage.total_tokens}"
            )
            return callback_response.model_dump()

        callback_response = APIResponse.failure_response(
            error=result.error or "Unknown error occurred",
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
            callback_url=str(request.callback_url) if request.callback_url else None,
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
