import logging

import openai
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.langfuse.langfuse import LangfuseTracer
from app.crud.credentials import get_provider_credential
from app.models import (
    CallbackResponse,
    Diagnostics,
    ResponsesAPIRequest,
    ResponseJobStatus,
    ResponsesSyncAPIRequest,
)
from app.services.response.jobs import start_job
from app.services.response.response import get_file_search_results
from app.services.response.callbacks import get_additional_data
from app.utils import (
    APIResponse,
    get_openai_client,
    handle_openai_error,
    load_description,
)


logger = logging.getLogger(__name__)
router = APIRouter(tags=["Responses"])


@router.post(
    "/responses",
    response_model=APIResponse[ResponseJobStatus],
    description=load_description("responses/create_async.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def responses(
    request: ResponsesAPIRequest,
    _session: SessionDep,
    _current_user: AuthContextDep,
):
    """Asynchronous endpoint that processes requests using Celery."""
    project_id, organization_id = (
        _current_user.project_.id,
        _current_user.organization_.id,
    )

    start_job(
        db=_session,
        request=request,
        project_id=project_id,
        organization_id=organization_id,
    )
    request_dict = request.model_dump()

    additional_data = get_additional_data(request_dict)

    response = ResponseJobStatus(
        status="processing",
        message="Your request is being processed. You will receive a callback once it's complete.",
        **additional_data,
    )
    return APIResponse.success_response(data=response)


@router.post(
    "/responses/sync",
    response_model=APIResponse[CallbackResponse],
    description=load_description("responses/create_sync.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def responses_sync(
    request: ResponsesSyncAPIRequest,
    _session: SessionDep,
    _current_user: AuthContextDep,
):
    """Synchronous endpoint for benchmarking OpenAI responses API with Langfuse tracing."""
    project_id, organization_id = (
        _current_user.project_.id,
        _current_user.organization_.id,
    )

    try:
        client = get_openai_client(_session, organization_id, project_id)
    except HTTPException as e:
        request_dict = request.model_dump()
        additional_data = get_additional_data(request_dict)
        return JSONResponse(
            status_code=e.status_code,
            content={
                "success": False,
                "data": additional_data if additional_data else None,
                "error": str(e.detail),
                "metadata": None,
            },
        )

    langfuse_credentials = get_provider_credential(
        session=_session,
        org_id=organization_id,
        provider="langfuse",
        project_id=project_id,
    )
    tracer = LangfuseTracer(
        credentials=langfuse_credentials,
        response_id=request.response_id,
    )

    tracer.start_trace(
        name="generate_response_sync", input={"question": request.question}
    )
    tracer.start_generation(
        name="openai_response",
        input={"question": request.question},
        metadata={"model": request.model, "temperature": request.temperature},
    )

    try:
        response = client.responses.create(
            model=request.model,
            previous_response_id=request.response_id,
            instructions=request.instructions,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": request.vector_store_ids,
                    "max_num_results": request.max_num_results,
                }
            ],
            temperature=request.temperature,
            input=[{"role": "user", "content": request.question}],
            include=["file_search_call.results"],
        )

        response_chunks = get_file_search_results(response)

        tracer.end_generation(
            output={
                "response_id": response.id,
                "message": response.output_text,
            },
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

        tracer.flush()
        logger.info(
            f"[response_sync] Successfully generated response: response_id={response.id}, project_id={project_id}"
        )

        request_dict = request.model_dump()
        additional_data = get_additional_data(request_dict)

        return APIResponse.success_response(
            data=CallbackResponse(
                status="success",
                response_id=response.id,
                message=response.output_text,
                chunks=response_chunks,
                diagnostics=Diagnostics(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                    model=response.model,
                ),
                **additional_data,
            )
        )
    except openai.OpenAIError as e:
        error_message = handle_openai_error(e)
        logger.error(
            f"[response_sync] OpenAI API error during response processing: {error_message}, project_id={project_id}",
            exc_info=True,
        )
        tracer.log_error(error_message, response_id=request.response_id)
        tracer.flush()

        request_dict = request.model_dump()
        # Create a custom error response with additional data in data field
        additional_data = get_additional_data(request_dict)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": additional_data if additional_data else None,
                "error": error_message,
                "metadata": None,
            },
        )
