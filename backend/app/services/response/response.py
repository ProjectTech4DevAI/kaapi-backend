import logging
from uuid import UUID

import openai
from openai import OpenAI
from openai.types.responses.response import Response
from fastapi import HTTPException
from sqlmodel import Session

from app.core.db import engine
from app.core.langfuse.langfuse import LangfuseTracer
from app.crud import (
    JobCrud,
    get_assistant_by_id,
    get_provider_credential,
    create_conversation,
    get_ancestor_id_from_response,
    get_conversation_by_ancestor_id,
)
from app.crud.project import is_tracing_enabled
from app.models import (
    CallbackResponse,
    Diagnostics,
    FileResultChunk,
    Assistant,
    JobStatus,
    JobUpdate,
    ResponsesAPIRequest,
    OpenAIConversationCreate,
    OpenAIConversation,
)
from app.utils import (
    APIResponse,
    get_openai_client,
    handle_openai_error,
    mask_string,
)

logger = logging.getLogger(__name__)


def get_file_search_results(response: Response) -> list[FileResultChunk]:
    """Extract file search results from a response."""
    results: list[FileResultChunk] = []
    for tool_call in response.output:
        if tool_call.type == "file_search_call":
            results.extend(
                FileResultChunk(
                    score=hit.score,
                    text=hit.text,
                    filename=getattr(hit, "filename", None),
                )
                for hit in tool_call.results
            )
    return results


def _build_callback_response(response: Response) -> CallbackResponse:
    """Build callback response with diagnostics and search results."""
    return CallbackResponse(
        status="success",
        response_id=response.id,
        message=response.output_text,
        diagnostics=Diagnostics(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
            model=response.model,
        ),
    )


def _fail_job(job_id: UUID, error_message: str) -> APIResponse:
    with Session(engine) as session:
        JobCrud(session=session).update(
            job_id=job_id,
            job_update=JobUpdate(
                status=JobStatus.FAILED,
                error_message=error_message,
            ),
        )
    return APIResponse.failure_response(error=error_message)


def generate_response(
    tracer: LangfuseTracer,
    client: OpenAI,
    assistant: Assistant,
    request: ResponsesAPIRequest,
    ancestor_id: str | None,
) -> tuple[Response | None, str | None]:
    """Generate a response using OpenAI and track with Langfuse."""
    response: Response | None = None
    error_message: str | None = None

    try:
        tracer.start_trace(
            name="generate_response_async",
            input={
                "question": request.question,
                "assistant_id": assistant.assistant_id,
            },
            metadata={"callback_url": request.callback_url},
            tags=[assistant.assistant_id],
        )
        tracer.start_generation(
            name="openai_response",
            input={"question": request.question},
            metadata={"model": assistant.model, "temperature": assistant.temperature},
        )

        params: dict = {
            "model": assistant.model,
            "instructions": assistant.instructions,
            "temperature": assistant.temperature,
            "input": [{"role": "user", "content": request.question}],
        }
        if ancestor_id:
            params["previous_response_id"] = ancestor_id

        if assistant.vector_store_ids:
            params["tools"] = [
                {
                    "type": "file_search",
                    "vector_store_ids": assistant.vector_store_ids,
                    "max_num_results": assistant.max_num_results,
                }
            ]
            params["include"] = ["file_search_call.results"]

        response = client.responses.create(**params)

        tracer.end_generation(
            output={"response_id": response.id, "message": response.output_text},
            usage={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.total_tokens,
                "unit": "TOKENS",
            },
            model=response.model,
        )
        tracer.update_trace(
            tags=[response.id],
            output={
                "status": "success",
                "message": response.output_text,
                "error": None,
            },
        )

    except openai.OpenAIError as e:
        error_message = handle_openai_error(e)
        logger.warning(
            f"[process_response_task] OpenAI API error: {error_message}",
            exc_info=True,
        )
        if tracer:
            tracer.log_error(error_message, response_id=request.response_id)

    finally:
        # Async path runs in a Celery worker; flush so spans aren't lost if the
        # worker is killed before the OTel batch processor's timer fires.
        tracer.flush()

    return response, error_message


def persist_conversation(
    response: Response,
    request: ResponsesAPIRequest,
    project_id: int,
    organization_id: int,
    job_id: UUID,
    assistant_id: str,
    latest_conversation: OpenAIConversation | None,
) -> None:
    """Persist conversation and mark job as successful."""
    with Session(engine) as session:
        ancestor_response_id = (
            latest_conversation.ancestor_response_id
            if latest_conversation
            else get_ancestor_id_from_response(
                session=session,
                current_response_id=response.id,
                previous_response_id=response.previous_response_id,
                project_id=project_id,
            )
        )

        create_conversation(
            session=session,
            conversation=OpenAIConversationCreate(
                response_id=response.id,
                previous_response_id=response.previous_response_id,
                ancestor_response_id=ancestor_response_id,
                user_question=request.question,
                response=response.output_text,
                model=response.model,
                assistant_id=assistant_id,
            ),
            project_id=project_id,
            organization_id=organization_id,
        )

        JobCrud(session=session).update(
            job_id=job_id,
            job_update=JobUpdate(status=JobStatus.SUCCESS),
        )


def process_response(
    request: ResponsesAPIRequest,
    project_id: int,
    organization_id: int,
    job_id: UUID,
    task_id: str,
    task_instance,
) -> APIResponse:
    assistant_id = request.assistant_id
    logger.info(
        f"[process_response_task] Generating response for "
        f"assistant_id={mask_string(assistant_id)}, "
        f"project_id={project_id}, task_id={task_id}, job_id={job_id}"
    )

    latest_conversation: OpenAIConversation | None = None

    try:
        with Session(engine) as session:
            JobCrud(session=session).update(
                job_id=job_id,
                job_update=JobUpdate(status=JobStatus.PROCESSING, task_id=task_id),
            )

            assistant = get_assistant_by_id(session, assistant_id, project_id)
            if not assistant:
                logger.warning(
                    f"[process_response_task] Assistant not found: "
                    f"assistant_id={mask_string(assistant_id)}, project_id={project_id}"
                )
                return _fail_job(job_id, "Assistant not found or not active")

            try:
                client = get_openai_client(session, organization_id, project_id)
            except HTTPException as e:
                return _fail_job(job_id, str(e.detail))

            langfuse_credentials = None
            if is_tracing_enabled(session=session, project_id=project_id):
                langfuse_credentials = get_provider_credential(
                    session=session,
                    org_id=organization_id,
                    provider="langfuse",
                    project_id=project_id,
                )

            ancestor_id = request.response_id
            if ancestor_id:
                latest_conversation = get_conversation_by_ancestor_id(
                    session,
                    ancestor_response_id=ancestor_id,
                    project_id=project_id,
                )
                if latest_conversation:
                    ancestor_id = latest_conversation.response_id

        tracer = LangfuseTracer(
            credentials=langfuse_credentials,
            response_id=request.response_id,
        )
        response, error_message = generate_response(
            tracer=tracer,
            client=client,
            assistant=assistant,
            request=request,
            ancestor_id=ancestor_id,
        )

        if response:
            persist_conversation(
                response,
                request,
                project_id,
                organization_id,
                job_id,
                assistant_id,
                latest_conversation,
            )
            return APIResponse.success_response(data=_build_callback_response(response))
        else:
            return _fail_job(job_id, error_message or "Unknown error")

    except Exception as e:
        logger.error(f"[process_response_task] Unexpected error: {e}", exc_info=True)
        return _fail_job(job_id, f"Unexpected error: {str(e)}")
