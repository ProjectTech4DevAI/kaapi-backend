import logging
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
from app.crud.llm import create_llm_call, update_llm_call_response
from app.models import JobStatus, JobType, JobUpdate, LLMCallRequest, Job
from app.models.llm.request import (
    ConfigBlob,
    LLMCallConfig,
    KaapiCompletionConfig,
    TextInput,
)
from app.models.llm.response import TextOutput
from app.services.llm.guardrails import (
    list_validators_config,
    run_guardrails_validation,
)
from app.services.llm.providers.registry import get_llm_provider
from app.services.llm.mappers import transform_kaapi_config_to_native
from app.services.llm.input_resolver import resolve_input, cleanup_temp_file

from app.utils import APIResponse, send_callback

logger = logging.getLogger(__name__)


def start_job(
    db: Session, request: LLMCallRequest, project_id: int, organization_id: int
) -> UUID:
    """Create an LLM job and schedule Celery task."""
    trace_id = correlation_id.get() or "N/A"
    job_crud = JobCrud(session=db)
    job = job_crud.create(job_type=JobType.LLM_API, trace_id=trace_id)

    # Explicitly flush to ensure job is persisted before Celery task starts
    db.flush()
    db.commit()

    logger.info(
        f"[start_job] Created job | job_id={job.id}, status={job.status}, project_id={project_id}"
    )

    try:
        task_id = start_high_priority_job(
            function_path="app.services.llm.jobs.execute_job",
            project_id=project_id,
            job_id=str(job.id),
            trace_id=trace_id,
            request_data=request.model_dump(mode="json"),
            organization_id=organization_id,
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
    job_id: UUID = UUID(job_id)

    config = request.config
    callback_response = None
    config_blob: ConfigBlob | None = None
    input_guardrails: list[dict] = []
    output_guardrails: list[dict] = []
    llm_call_id: UUID | None = None  # Track the LLM call record

    logger.info(
        f"[execute_job] Starting LLM job execution | job_id={job_id}, task_id={task_id}, "
    )

    try:
        with Session(engine) as session:
            # Update job status to PROCESSING
            job_crud = JobCrud(session=session)
            logger.info(f"[execute_job] Attempting to fetch job | job_id={job_id}")
            job = session.get(Job, job_id)
            if not job:
                # Log all jobs to see what's in the database
                from sqlmodel import select

                all_jobs = session.exec(
                    select(Job).order_by(Job.created_at.desc()).limit(5)
                ).all()
                logger.error(
                    f"[execute_job] Job not found! | job_id={job_id} | "
                    f"Recent jobs in DB: {[(j.id, j.status) for j in all_jobs]}"
                )
            else:
                logger.info(
                    f"[execute_job] Found job | job_id={job_id}, status={job.status}"
                )

            job_crud.update(
                job_id=job_id, job_update=JobUpdate(status=JobStatus.PROCESSING)
            )

            # if stored config, fetch blob from DB
            if config.is_stored_config:
                config_crud = ConfigVersionCrud(
                    session=session, project_id=project_id, config_id=config.id
                )

                # blob is dynamic, need to resolve to ConfigBlob format
                config_blob, error = resolve_config_blob(config_crud, config)

                if error:
                    callback_response = APIResponse.failure_response(
                        error=error,
                        metadata=request.request_metadata,
                    )
                    return handle_job_error(
                        job_id, request.callback_url, callback_response
                    )

            else:
                config_blob = config.blob

            if config_blob is not None:
                if config_blob.input_guardrails or config_blob.output_guardrails:
                    input_guardrails, output_guardrails = list_validators_config(
                        organization_id=organization_id,
                        project_id=project_id,
                        input_validator_configs=config_blob.input_guardrails,
                        output_validator_configs=config_blob.output_guardrails,
                    )

            if input_guardrails:
                if not isinstance(request.query.input, TextInput):
                    logger.info(
                        "[execute_job] Skipping input guardrails for non-text input. "
                        f"job_id={job_id}, input_type={getattr(request.query.input, 'type', type(request.query.input).__name__)}"
                    )
                else:
                    safe_input = run_guardrails_validation(
                        request.query.input.content.value,
                        input_guardrails,
                        job_id,
                        project_id,
                        organization_id,
                        suppress_pass_logs=True,
                    )

                    logger.info(
                        f"[execute_job] Input guardrail validation | success={safe_input['success']}."
                    )

                    if safe_input.get("bypassed"):
                        logger.info(
                            "[execute_job] Guardrails bypassed (service unavailable)"
                        )

                    elif safe_input["success"]:
                        request.query.input.content.value = safe_input["data"][
                            "safe_text"
                        ]
                    else:
                        # Update the text value with error message
                        request.query.input.content.value = safe_input["error"]

                        callback_response = APIResponse.failure_response(
                            error=safe_input["error"],
                            metadata=request.request_metadata,
                        )
                        return handle_job_error(
                            job_id, request.callback_url, callback_response
                        )
            user_sent_config_provider = ""

            try:
                # Transform Kaapi config to native config if needed (before getting provider)
                completion_config = config_blob.completion

                original_provider = (
                    config_blob.completion.provider
                )  # openai, google or prefixed

                if isinstance(completion_config, KaapiCompletionConfig):
                    completion_config, warnings = transform_kaapi_config_to_native(
                        completion_config
                    )

                    if request.request_metadata is None:
                        request.request_metadata = {}
                    request.request_metadata.setdefault("warnings", []).extend(warnings)
                else:
                    pass
            except Exception as e:
                callback_response = APIResponse.failure_response(
                    error=f"Error processing configuration: {str(e)}",
                    metadata=request.request_metadata,
                )
                return handle_job_error(job_id, request.callback_url, callback_response)

            # Create LLM call record before execution
            try:
                # Rebuild ConfigBlob with transformed native config
                resolved_config_blob = ConfigBlob(
                    completion=completion_config,
                    input_guardrails=config_blob.input_guardrails,
                    output_guardrails=config_blob.output_guardrails,
                )

                llm_call = create_llm_call(
                    session,
                    request=request,
                    job_id=job_id,
                    project_id=project_id,
                    organization_id=organization_id,
                    resolved_config=resolved_config_blob,
                    original_provider=original_provider,
                )
                llm_call_id = llm_call.id
                logger.info(
                    f"[execute_job] Created LLM call record | llm_call_id={llm_call_id}, job_id={job_id}"
                )
            except Exception as e:
                logger.error(
                    f"[execute_job] Failed to create LLM call record: {str(e)} | job_id={job_id}",
                    exc_info=True,
                )
                callback_response = APIResponse.failure_response(
                    error=f"Failed to create LLM call record: {str(e)}",
                    metadata=request.request_metadata,
                )
                return handle_job_error(job_id, request.callback_url, callback_response)

            try:
                provider_instance = get_llm_provider(
                    session=session,
                    provider_type=completion_config.provider,  # Now always native provider type i.e openai-native, google-native regardless
                    project_id=project_id,
                    organization_id=organization_id,
                )
            except ValueError as ve:
                callback_response = APIResponse.failure_response(
                    error=str(ve),
                    metadata=request.request_metadata,
                )
                return handle_job_error(job_id, request.callback_url, callback_response)

            langfuse_credentials = get_provider_credential(
                session=session,
                org_id=organization_id,
                project_id=project_id,
                provider="langfuse",
            )

        # Extract conversation_id for langfuse session grouping
        conversation_id = None
        if request.query.conversation and request.query.conversation.id:
            conversation_id = request.query.conversation.id

        # Resolve input (handles text, audio_base64, audio_url)
        resolved_input, resolve_error = resolve_input(request.query.input)
        if resolve_error:
            callback_response = APIResponse.failure_response(
                error=resolve_error,
                metadata=request.request_metadata,
            )
            return handle_job_error(job_id, request.callback_url, callback_response)

        # Apply Langfuse observability decorator to provider execute method
        decorated_execute = observe_llm_execution(
            credentials=langfuse_credentials,
            session_id=conversation_id,
        )(provider_instance.execute)

        try:
            response, error = decorated_execute(
                completion_config=completion_config,
                query=request.query,
                resolved_input=resolved_input,
                include_provider_raw_response=request.include_provider_raw_response,
            )
        finally:
            # Clean up temp files for audio inputs
            if resolved_input and resolved_input != request.query.input:
                cleanup_temp_file(resolved_input)

        if response:
            if output_guardrails:
                if not isinstance(response.response.output, TextOutput):
                    logger.info(
                        "[execute_job] Skipping output guardrails for non-text output. "
                        f"job_id={job_id}, output_type={getattr(response.response.output, 'type', type(response.response.output).__name__)}"
                    )
                else:
                    output_text = response.response.output.content.value
                    safe_output = run_guardrails_validation(
                        output_text,
                        output_guardrails,
                        job_id,
                        project_id,
                        organization_id,
                        suppress_pass_logs=True,
                    )

                    logger.info(
                        f"[execute_job] Output guardrail validation | success={safe_output['success']}."
                    )

                    if safe_output.get("bypassed"):
                        logger.info(
                            "[execute_job] Guardrails bypassed (service unavailable)"
                        )

                    elif safe_output["success"]:
                        response.response.output.content.value = safe_output["data"][
                            "safe_text"
                        ]

                        if safe_output["data"]["rephrase_needed"] == True:
                            callback_response = APIResponse.failure_response(
                                error=request.query.input,
                                metadata=request.request_metadata,
                            )
                            return handle_job_error(
                                job_id, request.callback_url, callback_response
                            )

                    else:
                        response.response.output.content.value = safe_output["error"]

                        callback_response = APIResponse.failure_response(
                            error=safe_output["error"],
                            metadata=request.request_metadata,
                        )
                        return handle_job_error(
                            job_id, request.callback_url, callback_response
                        )

            callback_response = APIResponse.success_response(
                data=response, metadata=request.request_metadata
            )
            if request.callback_url:
                send_callback(
                    callback_url=request.callback_url,
                    data=callback_response.model_dump(),
                )

            with Session(engine) as session:
                job_crud = JobCrud(session=session)

                # Update LLM call record with response data
                if llm_call_id:
                    try:
                        update_llm_call_response(
                            session,
                            llm_call_id=llm_call_id,
                            provider_response_id=response.response.provider_response_id,
                            content=response.response.output.model_dump(),
                            usage=response.usage.model_dump(),
                            conversation_id=response.response.conversation_id,
                        )
                        logger.info(
                            f"[execute_job] Updated LLM call record | llm_call_id={llm_call_id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[execute_job] Failed to update LLM call record: {str(e)} | llm_call_id={llm_call_id}",
                            exc_info=True,
                        )
                        # Don't fail the job if updating the record fails

                job_crud.update(
                    job_id=job_id, job_update=JobUpdate(status=JobStatus.SUCCESS)
                )
                logger.info(
                    f"[execute_job] Successfully completed LLM job | job_id={job_id}, "
                    f"provider_response_id={response.response.provider_response_id}, tokens={response.usage.total_tokens}"
                )
                return callback_response.model_dump()

        callback_response = APIResponse.failure_response(
            error=error or "Unknown error occurred",
            metadata=request.request_metadata,
        )
        return handle_job_error(job_id, request.callback_url, callback_response)

    except Exception as e:
        callback_response = APIResponse.failure_response(
            error=f"Unexpected error occurred",
            metadata=request.request_metadata,
        )
        logger.error(
            f"[execute_job] Unknown error occurred: {str(e)} | job_id={job_id}, task_id={task_id}",
            exc_info=True,
        )
        return handle_job_error(job_id, request.callback_url, callback_response)
