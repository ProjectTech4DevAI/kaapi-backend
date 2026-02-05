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
from app.models import JobStatus, JobType, JobUpdate, LLMCallRequest, LLMChainResponse
from app.models.llm.request import (
    ConfigBlob,
    KaapiCompletionConfig,
    LLMChainRequest,
    QueryParams,
    TextInput,
)
from app.services.llm.input_resolver import cleanup_temp_file, resolve_input
from app.services.llm.jobs import handle_job_error, resolve_config_blob
from app.services.llm.mappers import transform_kaapi_config_to_native
from app.services.llm.providers.registry import get_llm_provider
from app.utils import APIResponse, send_callback


logger = logging.getLogger(__name__)


def start_chain_job(
    db: Session,
    request: LLMChainRequest,
    project_id: int,
    organization_id: int,
) -> UUID:
    """Create an LLM chain job and schedule Celery task."""
    trace_id = correlation_id.get() or "N/A"
    job_crud = JobCrud(session=db)
    job = job_crud.create(job_type=JobType.LLM_CHAIN, trace_id=trace_id)

    db.flush()
    db.commit()

    logger.info(
        f"[start_chain_job] Created chain job | job_id={job.id}, "
        f"blocks={len(request.blocks)}, project_id={project_id}"
    )

    try:
        task_id = start_high_priority_job(
            function_path="app.services.llm.chain_executor.execute_chain_job",
            project_id=project_id,
            job_id=str(job.id),
            trace_id=trace_id,
            request_data=request.model_dump(mode="json"),
            organization_id=organization_id,
        )
    except Exception as e:
        logger.error(
            f"[start_chain_job] Error starting Celery task: {str(e)} | job_id={job.id}",
            exc_info=True,
        )
        job_crud.update(
            job_id=job.id,
            job_update=JobUpdate(status=JobStatus.FAILED, error_message=str(e)),
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error while starting chain execution",
        )

    logger.info(
        f"[start_chain_job] Chain job scheduled | job_id={job.id}, task_id={task_id}"
    )
    return job.id


def _interpolate_template(text: str, config_blob: ConfigBlob) -> str:
    if config_blob.prompt_template:
        return config_blob.prompt_template.template.replace("{{input}}", text)
    return text


def execute_chain_job(
    request_data: dict,
    project_id: int,
    organization_id: int,
    job_id: str,
    task_id: str,
    task_instance,
) -> dict:
    """Celery task to process an LLM chain request asynchronously."""
    request = LLMChainRequest(**request_data)
    job_id: UUID = UUID(job_id)
    callback_url = str(request.callback_url) if request.callback_url else None

    logger.info(
        f"[execute_chain_job] Starting chain | job_id={job_id}, "
        f"blocks={len(request.blocks)}, task_id={task_id}"
    )

    try:
        with Session(engine) as session:
            JobCrud(session=session).update(
                job_id=job_id,
                job_update=JobUpdate(status=JobStatus.PROCESSING),
            )

        previous_output: str | None = None
        last_response = None

        for block_idx, block in enumerate(request.blocks):
            is_last = block_idx == len(request.blocks) - 1

            block_config = block.config

            logger.info(f"[BLOCK CONFIG] ===> {block_config}")

            logger.info(
                f"[execute_chain_job] Executing block {block_idx}/{len(request.blocks) - 1} "
                f"| job_id={job_id}"
            )

            if block_idx > 0 and not previous_output:
                callback_response = APIResponse.failure_response(
                    error=f"Block {block_idx - 1} returned empty output, cannot continue chain",
                    metadata=block.request_metadata,
                )
                return handle_job_error(job_id, callback_url, callback_response)

            with Session(engine) as session:
                config_blob: ConfigBlob | None = None

                if block_config.is_stored_config:
                    config_crud = ConfigVersionCrud(
                        session=session,
                        project_id=project_id,
                        config_id=block_config.id,
                    )
                    config_blob, error = resolve_config_blob(config_crud, block_config)
                    if error:
                        callback_response = APIResponse.failure_response(
                            error=f"Block {block_idx}: {error}",
                            metadata=block.request_metadata,
                        )
                        return handle_job_error(job_id, callback_url, callback_response)
                else:
                    config_blob = block_config.blob

                completion_config = config_blob.completion
                original_provider = completion_config.provider

                if isinstance(completion_config, KaapiCompletionConfig):
                    try:
                        completion_config, warnings = transform_kaapi_config_to_native(
                            completion_config
                        )
                    except Exception as e:
                        callback_response = APIResponse.failure_response(
                            error=f"Block {block_idx}: Config transformation error: {str(e)}",
                            metadata=block.request_metadata,
                        )
                        return handle_job_error(job_id, callback_url, callback_response)

                if block_idx == 0:
                    block_query = request.query
                    if config_blob.prompt_template and isinstance(
                        request.query.input, TextInput
                    ):
                        interpolated = _interpolate_template(
                            request.query.input.content, config_blob
                        )
                        block_query = QueryParams(input=interpolated)
                else:
                    block_input = _interpolate_template(previous_output, config_blob)
                    block_query = QueryParams(input=block_input)

                resolved_config_blob = ConfigBlob(completion=completion_config)
                synthetic_request = LLMCallRequest(
                    query=block_query,
                    config=block_config,
                    request_metadata=block.request_metadata,
                )

                try:
                    llm_call = create_llm_call(
                        session,
                        request=synthetic_request,
                        job_id=job_id,
                        project_id=project_id,
                        organization_id=organization_id,
                        resolved_config=resolved_config_blob,
                        original_provider=original_provider,
                    )
                    llm_call_id = llm_call.id
                except Exception as e:
                    logger.error(
                        f"[execute_chain_job] Failed to create LLM call record "
                        f"for block {block_idx}: {str(e)}",
                        exc_info=True,
                    )
                    callback_response = APIResponse.failure_response(
                        error=f"Block {block_idx}: Failed to create LLM call record",
                        metadata=block.request_metadata,
                    )
                    return handle_job_error(job_id, callback_url, callback_response)

                try:
                    provider_instance = get_llm_provider(
                        session=session,
                        provider_type=completion_config.provider,
                        project_id=project_id,
                        organization_id=organization_id,
                    )
                except ValueError as ve:
                    callback_response = APIResponse.failure_response(
                        error=f"Block {block_idx}: {str(ve)}",
                        metadata=block.request_metadata,
                    )
                    return handle_job_error(job_id, callback_url, callback_response)

                langfuse_credentials = get_provider_credential(
                    session=session,
                    org_id=organization_id,
                    project_id=project_id,
                    provider="langfuse",
                )

            conversation_id = None
            if block_query.conversation and block_query.conversation.id:
                conversation_id = block_query.conversation.id

            resolved_input, resolve_error = resolve_input(block_query.input)
            if resolve_error:
                callback_response = APIResponse.failure_response(
                    error=f"Block {block_idx}: {resolve_error}",
                    metadata=block.request_metadata,
                )
                return handle_job_error(job_id, callback_url, callback_response)

            decorated_execute = observe_llm_execution(
                credentials=langfuse_credentials,
                session_id=conversation_id,
            )(provider_instance.execute)

            try:
                response, error = decorated_execute(
                    completion_config=completion_config,
                    query=block_query,
                    resolved_input=resolved_input,
                    include_provider_raw_response=block.include_provider_raw_response,
                )
            finally:
                if resolved_input and resolved_input != block_query.input:
                    cleanup_temp_file(resolved_input)

            if not response:
                callback_response = APIResponse.failure_response(
                    error=f"Block {block_idx}: {error or 'Unknown error'}",
                    metadata=block.request_metadata,
                )
                return handle_job_error(job_id, callback_url, callback_response)

            with Session(engine) as session:
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
                    logger.error(
                        f"[execute_chain_job] Failed to update LLM call record "
                        f"for block {block_idx}: {str(e)}",
                        exc_info=True,
                    )

            previous_output = response.response.output.text
            last_response = response

            logger.info(
                f"[execute_chain_job] Block {block_idx} completed | job_id={job_id}, "
                f"provider={response.response.provider}, model={response.response.model}, "
                f"tokens={response.usage.total_tokens}"
            )

            if not is_last and block.intermediate_callback and callback_url:
                send_callback(
                    callback_url=callback_url,
                    data=APIResponse.success_response(
                        data={
                            "type": "intermediary",
                            "block_index": block_idx + 1,
                            "blocks_total": len(request.blocks),
                            "response": response.model_dump(),
                        },
                        metadata=block.request_metadata,
                    ).model_dump(),
                )

        chain_response = LLMChainResponse(
            response=last_response,
            # blocks_executed=len(request.blocks),
        )

        callback_response = APIResponse.success_response(
            data=chain_response, metadata=block.request_metadata
        )

        if callback_url:
            send_callback(
                callback_url=callback_url,
                data=callback_response.model_dump(),
            )

        with Session(engine) as session:
            JobCrud(session=session).update(
                job_id=job_id,
                job_update=JobUpdate(status=JobStatus.SUCCESS),
            )

        logger.info(
            f"[execute_chain_job] Chain completed | job_id={job_id}, "
            f"blocks_executed={len(request.blocks)}"
        )

        return callback_response.model_dump()

    except Exception as e:
        logger.error(
            f"[execute_chain_job] Unexpected error: {str(e)} | job_id={job_id}, task_id={task_id}",
            exc_info=True,
        )
        callback_response = APIResponse.failure_response(
            error="Unexpected error occurred during chain execution",
            metadata=block.request_metadata,
        )
        return handle_job_error(job_id, callback_url, callback_response)
