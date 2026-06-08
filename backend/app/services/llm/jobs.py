import base64
import json
import logging
import time
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
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
from app.crud.model_config import validate_blob_model_or_raise
from app.crud.jobs import JobCrud
from app.crud.llm import (
    create_llm_call,
    serialize_input,
    update_llm_call_input,
    update_llm_call_response,
    save_rephrase_guardrail_call,
)
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
    TextContent,
    TextInput,
)
from app.core.cloud.storage import get_cloud_storage
from app.core.storage_utils import upload_audio_bytes_to_s3
from app.models.llm.response import (
    AudioOutput,
    LLMCallResponse,
    LLMResponse,
    TextOutput,
    Usage,
)
from app.services.llm.chain.types import BlockResult
from app.services.llm.guardrails import (
    list_validators_config,
    run_guardrails_validation,
)
from app.services.llm.mappers import transform_kaapi_config_to_native
from app.services.llm.providers.registry import get_llm_provider
from app.utils import (
    APIResponse,
    download_audio_bytes,
    get_webhook_secret,
    resolve_input,
    send_callback,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def _set_traceability_attributes(
    span: trace.Span,
    *,
    job_id: UUID | str | None = None,
    llm_call_id: UUID | str | None = None,
    chain_id: UUID | str | None = None,
    trace_id: str | None = None,
    project_id: int | None = None,
    organization_id: int | None = None,
    task_id: str | None = None,
) -> None:
    if job_id is not None:
        span.set_attribute("llm.job_id", str(job_id))
    if llm_call_id is not None:
        span.set_attribute("llm.call_id", str(llm_call_id))
    if chain_id is not None:
        span.set_attribute("llm.chain_id", str(chain_id))
    if trace_id is not None:
        span.set_attribute("kaapi.trace_id", trace_id)
    if project_id is not None:
        span.set_attribute("kaapi.project_id", project_id)
    if organization_id is not None:
        span.set_attribute("kaapi.organization_id", organization_id)
    if task_id is not None:
        span.set_attribute("celery.task_id", task_id)


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
    if not request.config.is_stored_config and request.config.blob:
        validate_blob_model_or_raise(db, request.config.blob)

    with log_context(
        tag="llm-call",
        lifecycle="llm.call.start_job",
        project_id=project_id,
        organization_id=organization_id,
    ), tracer.start_as_current_span("llm.start_job") as span:
        trace_id = correlation_id.get() or "N/A"
        _set_traceability_attributes(
            span,
            project_id=project_id,
            organization_id=organization_id,
            trace_id=trace_id,
        )

        job_crud = JobCrud(session=db)
        job = job_crud.create(
            job_type=JobType.LLM_API, trace_id=trace_id, project_id=project_id
        )
        _set_traceability_attributes(span, job_id=job.id)

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

        _set_traceability_attributes(span, task_id=str(task_id))
        logger.info(
            f"[start_job] Job scheduled for LLM call | job_id={job.id}, project_id={project_id}, task_id={task_id}"
        )
        return job.id


def start_chain_job(
    db: Session, request: LLMChainRequest, project_id: int, organization_id: int
) -> UUID:
    """Create an LLM Chain job and schedule Celery task."""
    for block in request.blocks:
        if not block.config.is_stored_config and block.config.blob:
            validate_blob_model_or_raise(db, block.config.blob)

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
    ), tracer.start_as_current_span("llm.chain.start_job") as span:
        _set_traceability_attributes(
            span,
            job_id=job.id,
            trace_id=trace_id,
            project_id=project_id,
            organization_id=organization_id,
        )
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
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
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

        _set_traceability_attributes(span, task_id=str(task_id))
        logger.info(
            f"[start_chain_job] Job scheduled for LLM chain job | job_id={job.id}, project_id={project_id}, task_id={task_id}"
        )
        return job.id


def handle_job_error(
    job_id: UUID,
    callback_url: str | None,
    callback_response: APIResponse,
    organization_id: int | None = None,
    project_id: int | None = None,
    chain_id: UUID | None = None,
) -> dict:
    """Handle job failure uniformly — send callback and update DB."""
    with Session(engine) as session:
        JobCrud(session=session).update(
            job_id=job_id,
            job_update=JobUpdate(
                status=JobStatus.FAILED,
                error_message=callback_response.error,
            ),
        )
        if chain_id:
            try:
                update_llm_chain_status(
                    session,
                    chain_id=chain_id,
                    status=ChainStatus.FAILED,
                    error=callback_response.error,
                )
            except Exception as update_err:
                logger.error(
                    f"[handle_job_error] Failed to update chain status: {update_err} | "
                    f"chain_id={chain_id}",
                    exc_info=True,
                )

    if callback_url:
        webhook_secret = get_webhook_secret(project_id, organization_id)
        with tracer.start_as_current_span("llm.send_callback") as cb_span:
            cb_span.set_attribute("callback.url", callback_url)
            cb_span.set_attribute("callback.status", "failure")
            _set_traceability_attributes(
                cb_span,
                job_id=job_id,
                trace_id=correlation_id.get(),
                project_id=project_id,
                organization_id=organization_id,
            )
            send_callback(
                callback_url=callback_url,
                data=callback_response.model_dump(),
                webhook_secret=webhook_secret,
            )

    return callback_response.model_dump()


@contextmanager
def resolved_input_context(
    query_input: TextInput | AudioInput | ImageInput | PDFInput | list,
):
    """Resolve query input. Audio inputs return AudioRef (in-memory);
    providers materialize a temp file via ``audio_ref.to_path()`` only if
    their SDK needs one, and clean it up themselves.
    """
    resolved_input, error = resolve_input(query_input)
    if error:
        raise ValueError(error)
    yield resolved_input


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
        blob = ConfigBlob(**config_version.config_blob)
    except (TypeError, ValueError) as e:
        return None, f"Stored configuration blob is invalid: {str(e)}"
    except Exception:
        logger.error(
            f"[resolve_config_blob] Unexpected error parsing config blob | "
            f"config_id={config.id}, version={config.version}",
            exc_info=True,
        )
        return None, "Unexpected error occurred while parsing stored configuration"

    try:
        validate_blob_model_or_raise(config_crud.session, blob)
    except HTTPException as e:
        return None, e.detail

    return blob, None


def apply_input_guardrails(
    *,
    config_blob: ConfigBlob | None,
    query: QueryParams,
    job_id: UUID,
    project_id: int,
    organization_id: int,
) -> tuple[QueryParams, str | None, str | None]:
    """Apply input guardrails from a config_blob. Shared with llm-call and llm-chain.

    Returns (query, error, guardrail_direct_response) where:
    - error is set when guardrails hard-block the request
    - guardrail_direct_response is set when rephrase_needed=True and the safe_text
      should be returned directly to the user without hitting the LLM
    """
    if not config_blob or not config_blob.input_guardrails:
        return query, None, None

    if not isinstance(query.input, TextInput):
        logger.info(
            f"[apply_input_guardrails] Skipping for non-text input. "
            f"job_id={job_id}, "
            f"input_type={getattr(query.input, 'type', type(query.input).__name__)}"
        )
        return query, None, None

    input_guardrails, _ = list_validators_config(
        organization_id=organization_id,
        project_id=project_id,
        input_validator_configs=config_blob.input_guardrails,
        output_validator_configs=None,
    )

    if not input_guardrails:
        return query, None, None

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
        return query, None, None

    if safe["success"]:
        safe_text = safe["data"]["safe_text"]
        if safe["data"].get("rephrase_needed"):
            logger.info(
                f"[apply_input_guardrails] rephrase_needed=True, returning safe_text directly | job_id={job_id}"
            )
            return query, None, safe_text
        query.input.content.value = safe_text
        return query, None, None

    return query, safe["error"], None


def apply_output_guardrails(
    *,
    config_blob: ConfigBlob | None,
    result: BlockResult,
    job_id: UUID,
    project_id: int,
    organization_id: int,
    input_text: str | None = None,
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

    llm_output = result.response.response.output.content.value
    safe = run_guardrails_validation(
        input_text or "",
        output_guardrails,
        job_id,
        project_id,
        organization_id,
        suppress_pass_logs=True,
        output_text=llm_output,
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
    trace_id = correlation_id.get()

    try:
        with Session(engine) as session:
            with tracer.start_as_current_span("llm.resolve_config") as cfg_span:
                _set_traceability_attributes(
                    cfg_span,
                    job_id=job_id,
                    chain_id=chain_id,
                    trace_id=trace_id,
                    project_id=project_id,
                    organization_id=organization_id,
                )
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
                    try:
                        validate_blob_model_or_raise(session, config_blob)
                    except HTTPException as e:
                        cfg_span.set_status(
                            trace.Status(trace.StatusCode.ERROR, e.detail)
                        )
                        return BlockResult(error=e.detail)

            original_input_value = (
                query.input.content.value
                if isinstance(query.input, TextInput)
                else None
            )

            if config_blob.prompt_template and isinstance(query.input, TextInput):
                template = config_blob.prompt_template.template
                interpolated = template.replace("{{input}}", query.input.content.value)
                query.input.content.value = interpolated

            # if client_llm_url_present  proxy route

            with tracer.start_as_current_span("llm.guardrails.input") as guard_span:
                _set_traceability_attributes(
                    guard_span,
                    job_id=job_id,
                    chain_id=chain_id,
                    trace_id=trace_id,
                    project_id=project_id,
                    organization_id=organization_id,
                )
                if config_blob.params.client_llm_url:
                    # apply input guardrails directly
                    (
                        query,
                        input_error,
                        guardrail_direct_response,
                    ) = apply_input_guardrails(
                        config_blob=config_blob,
                        query=query,
                        job_id=job_id,
                        project_id=project_id,
                        organization_id=organization_id,
                    )
                    # get safe text
                    # make direct call to the client_llm_url
                    # bypass further execution

                query, input_error, guardrail_direct_response = apply_input_guardrails(
                    config_blob=config_blob,
                    query=query,
                    job_id=job_id,
                    project_id=project_id,
                    organization_id=organization_id,
                )
                if guardrail_direct_response is not None:
                    guardrail_usage = Usage(
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                    )
                    llm_response = LLMCallResponse(
                        response=LLMResponse(
                            provider_response_id=str(job_id),
                            provider=str(config_blob.completion.provider),
                            model=str(config_blob.completion.params.get("model") or ""),
                            output=TextOutput(
                                content=TextContent(value=guardrail_direct_response)
                            ),
                        ),
                        usage=guardrail_usage,
                    )
                    if original_input_value is not None:
                        query.input.content.value = original_input_value
                    llm_call_id = save_rephrase_guardrail_call(
                        session=session,
                        query=query,
                        config=config,
                        request_metadata=request_metadata,
                        config_blob=config_blob,
                        guardrail_direct_response=guardrail_direct_response,
                        job_id=job_id,
                        project_id=project_id,
                        organization_id=organization_id,
                        chain_id=chain_id,
                    )
                    return BlockResult(
                        response=llm_response,
                        usage=guardrail_usage,
                        metadata=request_metadata,
                        llm_call_id=llm_call_id,
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
                _set_traceability_attributes(
                    create_span,
                    job_id=job_id,
                    chain_id=chain_id,
                    trace_id=trace_id,
                    project_id=project_id,
                    organization_id=organization_id,
                )
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
                    _set_traceability_attributes(create_span, llm_call_id=llm_call_id)
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

            # Upload STT input audio to S3 and overwrite llm_call.input with the URI.
            # Failures are non-fatal: the job proceeds and the provider still gets the original input.
            if (
                isinstance(query.input, AudioInput)
                and query.input.content.format in ("base64", "url")
                and llm_call_id
            ):
                try:
                    if query.input.content.format == "url":
                        stt_bytes, dl_error = download_audio_bytes(
                            query.input.content.value
                        )
                        if dl_error or not stt_bytes:
                            raise ValueError(dl_error or "Empty audio bytes from URL")
                        # Rewrite to base64 in-place so the provider resolve path
                        # reuses these bytes instead of issuing a second HTTP download.
                        query.input.content.value = base64.b64encode(stt_bytes).decode()
                        query.input.content.format = "base64"
                    else:
                        stt_bytes = base64.b64decode(query.input.content.value)

                    storage = get_cloud_storage(session, project_id)
                    subfolder_path = f"orgs/{organization_id}/{project_id}/audio/stt"
                    s3_url = upload_audio_bytes_to_s3(
                        storage,
                        stt_bytes,
                        llm_call_id,
                        query.input.content.mime_type,
                        subfolder_path,
                    )
                    if s3_url:
                        stt_input_record = json.dumps(
                            {
                                "type": "audio",
                                "format": "uri",
                                "mime_type": query.input.content.mime_type,
                                "size_bytes": len(stt_bytes),
                                "uri": s3_url,
                            }
                        )
                        update_llm_call_input(session, llm_call_id, stt_input_record)
                        logger.info(
                            f"[execute_llm_call] STT audio uploaded to S3 | llm_call_id={llm_call_id}"
                        )
                    else:
                        logger.warning(
                            f"[execute_llm_call] STT S3 upload failed | llm_call_id={llm_call_id}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[execute_llm_call] STT S3 upload error, continuing: {e} | llm_call_id={llm_call_id}",
                        exc_info=True,
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
            _set_traceability_attributes(
                ai_span,
                job_id=job_id,
                llm_call_id=llm_call_id,
                chain_id=chain_id,
                trace_id=trace_id,
                project_id=project_id,
                organization_id=organization_id,
            )
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
                        _set_traceability_attributes(
                            provider_span,
                            job_id=job_id,
                            llm_call_id=llm_call_id,
                            chain_id=chain_id,
                            trace_id=trace_id,
                            project_id=project_id,
                            organization_id=organization_id,
                        )
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
            else:
                ai_span.set_status(
                    trace.Status(trace.StatusCode.ERROR, error or "Unknown error")
                )

        if response:
            # db_content is what gets persisted — URI-only for TTS to avoid storing
            # large base64 payloads. The in-memory response keeps base64 + uri field
            # so existing clients continue to receive base64 unchanged.
            db_content = (
                response.response.output.model_dump()
                if response.response.output
                else None
            )

            tts_output = response.response.output
            if (
                isinstance(tts_output, AudioOutput)
                and tts_output.content.format == "base64"
                and llm_call_id
            ):
                try:
                    with Session(engine) as s3_session:
                        storage = get_cloud_storage(s3_session, project_id)
                    tts_bytes = base64.b64decode(tts_output.content.value)
                    subfolder_path = f"orgs/{organization_id}/{project_id}/audio/tts"
                    s3_url = upload_audio_bytes_to_s3(
                        storage,
                        tts_bytes,
                        llm_call_id,
                        tts_output.content.mime_type,
                        subfolder_path,
                    )
                    if s3_url:
                        # Keep base64 in the response object for backward-compatible clients.
                        # Set uri so execute_job can swap it for a presigned URL.
                        tts_output.content.uri = s3_url
                        # Store only the URI in the DB — not the full base64.
                        db_content = {
                            "type": "audio",
                            "content": {
                                "format": "uri",
                                "value": s3_url,
                                "mime_type": tts_output.content.mime_type,
                            },
                        }
                        logger.info(
                            f"[execute_llm_call] TTS audio uploaded to S3 | llm_call_id={llm_call_id}"
                        )
                    else:
                        logger.warning(
                            f"[execute_llm_call] TTS S3 upload failed, keeping base64 | llm_call_id={llm_call_id}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[execute_llm_call] TTS S3 upload error, keeping base64: {e} | llm_call_id={llm_call_id}",
                        exc_info=True,
                    )

            with Session(engine) as session:
                if llm_call_id:
                    with tracer.start_as_current_span(
                        "llm.update_call_record"
                    ) as update_span:
                        _set_traceability_attributes(
                            update_span,
                            job_id=job_id,
                            llm_call_id=llm_call_id,
                            chain_id=chain_id,
                            trace_id=trace_id,
                            project_id=project_id,
                            organization_id=organization_id,
                        )
                        try:
                            update_llm_call_response(
                                session,
                                llm_call_id=llm_call_id,
                                provider_response_id=response.response.provider_response_id,
                                content=db_content,
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
                _set_traceability_attributes(
                    out_guard_span,
                    job_id=job_id,
                    llm_call_id=llm_call_id,
                    chain_id=chain_id,
                    trace_id=trace_id,
                    project_id=project_id,
                    organization_id=organization_id,
                )
                result, output_error = apply_output_guardrails(
                    config_blob=config_blob,
                    result=result,
                    job_id=job_id,
                    project_id=project_id,
                    organization_id=organization_id,
                    input_text=original_input_value,
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

    except (Timeout, SoftTimeLimitExceeded):
        raise
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
        _set_traceability_attributes(
            trace.get_current_span(),
            job_id=job_uuid,
            trace_id=correlation_id.get(),
            project_id=project_id,
            organization_id=organization_id,
            task_id=task_id,
        )
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
                # Swap the s3:// URI in content.uri for a short-lived presigned URL.
                # content.value (base64) is untouched — existing clients keep working.
                # On failure, clear uri so clients don't receive a raw s3:// address.
                if result.response:
                    tts_out = result.response.response.output
                    if isinstance(tts_out, AudioOutput) and tts_out.content.uri:
                        try:
                            with Session(engine) as s3_session:
                                storage = get_cloud_storage(s3_session, project_id)
                            tts_out.content.uri = storage.get_signed_url(
                                tts_out.content.uri, expires_in=3600
                            )
                        except Exception as e:
                            logger.warning(
                                f"[execute_job] Failed to generate presigned URL: {e} | job_id={job_uuid}",
                                exc_info=True,
                            )
                            tts_out.content.uri = None

                callback_response = APIResponse.success_response(
                    data=result.response, metadata=result.metadata
                )
                if callback_url_str:
                    webhook_secret = get_webhook_secret(project_id, organization_id)
                    with tracer.start_as_current_span("llm.send_callback") as cb_span:
                        cb_span.set_attribute("callback.url", callback_url_str)
                        cb_span.set_attribute("callback.status", "success")
                        _set_traceability_attributes(
                            cb_span,
                            job_id=job_uuid,
                            trace_id=correlation_id.get(),
                            project_id=project_id,
                            organization_id=organization_id,
                            task_id=task_id,
                        )
                        send_callback(
                            callback_url=callback_url_str,
                            data=callback_response.model_dump(),
                            webhook_secret=webhook_secret,
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

            callback_response = APIResponse.failure_response(
                error=result.error or "Unknown error occurred",
                metadata=request.request_metadata,
            )
            return handle_job_error(
                job_uuid,
                callback_url_str,
                callback_response,
                organization_id=organization_id,
                project_id=project_id,
            )

        except (Timeout, SoftTimeLimitExceeded):
            logger.warning(
                f"[execute_job] LLM job timed out | job_id={job_uuid}, task_id={task_id}"
            )
            callback_response = APIResponse.failure_response(
                error="Task exceeded soft time limit",
                metadata=request.request_metadata,
            )
            handle_job_error(
                job_uuid,
                callback_url_str,
                callback_response,
                organization_id=organization_id,
                project_id=project_id,
            )
            raise

        except Exception as e:
            callback_response = APIResponse.failure_response(
                error="Unexpected error occurred",
                metadata=request.request_metadata,
            )
            logger.error(
                f"[execute_job] Unexpected error: {str(e)} | job_id={job_uuid}, task_id={task_id}",
                exc_info=True,
            )
            return handle_job_error(
                job_uuid,
                callback_url_str,
                callback_response,
                organization_id=organization_id,
                project_id=project_id,
            )
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
        _set_traceability_attributes(
            trace.get_current_span(),
            job_id=job_uuid,
            trace_id=correlation_id.get(),
            project_id=project_id,
            organization_id=organization_id,
            task_id=task_id,
        )
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
                _set_traceability_attributes(
                    trace.get_current_span(), chain_id=chain_uuid
                )

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

        except (Timeout, SoftTimeLimitExceeded) as err:
            logger.warning(
                f"[execute_chain_job] Chain job timed out | job_id={job_uuid}, task_id={task_id}"
            )

            callback_response = APIResponse.failure_response(
                error="Task exceeded soft time limit",
                metadata=request.request_metadata,
            )
            handle_job_error(
                job_uuid,
                callback_url_str,
                callback_response,
                organization_id=organization_id,
                project_id=project_id,
                chain_id=chain_uuid,
            )
            raise

        except Exception as e:
            logger.error(
                f"[execute_chain_job] Failed: {e} | job_id={job_uuid}",
                exc_info=True,
            )

            callback_response = APIResponse.failure_response(
                error="Unexpected error occurred",
                metadata=request.request_metadata,
            )
            return handle_job_error(
                job_uuid,
                callback_url_str,
                callback_response,
                organization_id=organization_id,
                project_id=project_id,
                chain_id=chain_uuid,
            )
        finally:
            # Ensure task spans are pushed promptly so Sentry dashboards update faster.
            flush_telemetry()
