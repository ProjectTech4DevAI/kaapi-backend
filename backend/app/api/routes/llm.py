import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.jobs import JobCrud
from app.crud.llm import get_llm_calls_by_job_id
from app.models import (
    LLMCallRequest,
    LLMCallResponse,
    LLMJobImmediatePublic,
    LLMJobPublic,
    JobStatus,
)
from app.models.llm.response import LLMResponse, Usage
from app.services.llm.jobs import start_job
from app.utils import APIResponse, validate_callback_url, load_description


logger = logging.getLogger(__name__)

router = APIRouter(tags=["LLM"])
llm_callback_router = APIRouter()


@llm_callback_router.post(
    "{$callback_url}",
    name="llm_callback",
)
def llm_callback_notification(body: APIResponse[LLMCallResponse]):
    """
    Callback endpoint specification for LLM call completion.

    The callback will receive:
    - On success: APIResponse with success=True and data containing LLMCallResponse
    - On failure: APIResponse with success=False and error message
    - metadata field will always be included if provided in the request
    """
    ...


@router.post(
    "/llm/call",
    description=load_description("llm/llm_call.md"),
    response_model=APIResponse[LLMJobImmediatePublic],
    callbacks=llm_callback_router.routes,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def llm_call(
    _current_user: AuthContextDep, session: SessionDep, request: LLMCallRequest
):
    """
    Endpoint to initiate an LLM call as a background job.
    Returns job information for polling.
    """
    project_id = _current_user.project_.id
    organization_id = _current_user.organization_.id

    if request.callback_url:
        validate_callback_url(str(request.callback_url))

    job_id = start_job(
        db=session,
        request=request,
        project_id=project_id,
        organization_id=organization_id,
    )

    # Fetch job details to return immediate response
    job_crud = JobCrud(session=session)
    job = job_crud.get(job_id=job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if request.callback_url:
        message = "Your response is being generated and will be delivered via callback."
    else:
        message = "Your response is being generated"

    job_response = LLMJobImmediatePublic(
        job_id=job.id,
        status=job.status.value,
        message=message,
        job_inserted_at=job.created_at,
        job_updated_at=job.updated_at,
    )

    return APIResponse.success_response(data=job_response)


@router.get(
    "/llm/call/{job_id}",
    description=load_description("llm/get_llm_call.md"),
    response_model=APIResponse[LLMJobPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_llm_call_status(
    _current_user: AuthContextDep,
    session: SessionDep,
    job_id: UUID,
) -> APIResponse[LLMJobPublic]:
    """
    Poll for LLM call job status and results.
    Returns job information with nested LLM response when complete.
    """
    job_crud = JobCrud(session=session)
    job = job_crud.get(job_id=job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    llm_call_response = None
    if job.status.value == JobStatus.SUCCESS:
        llm_calls = get_llm_calls_by_job_id(
            session=session, job_id=job_id, project_id=_current_user.project_.id
        )

        if llm_calls:
            # Get the first LLM call from the list which will be the only call for the job id
            # since we initially won't be using this endpoint for llm chains
            llm_call = llm_calls[0]

            llm_response = LLMResponse(
                provider_response_id=llm_call.provider_response_id or "",
                conversation_id=llm_call.conversation_id,
                provider=llm_call.provider,
                model=llm_call.model,
                output=llm_call.content,
            )

            if not llm_call.usage:
                raise HTTPException(
                    status_code=500,
                    detail="Completed LLM job is missing usage data",
                )

            llm_call_response = LLMCallResponse(
                response=llm_response,
                usage=Usage(**llm_call.usage),
                provider_raw_response=None,
            )

    job_response = LLMJobPublic(
        job_id=job.id,
        status=job.status.value,
        llm_response=llm_call_response,
        error_message=job.error_message,
    )

    return APIResponse.success_response(data=job_response)
