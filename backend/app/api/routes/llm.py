import logging

from fastapi import APIRouter, Depends

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import LLMCallRequest, LLMCallResponse, Message
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
    response_model=APIResponse[Message],
    callbacks=llm_callback_router.routes,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def llm_call(
    _current_user: AuthContextDep, _session: SessionDep, request: LLMCallRequest
):
    """
    Endpoint to initiate an LLM call as a background job.
    """
    project_id = _current_user.project_.id
    organization_id = _current_user.organization_.id

    if request.callback_url:
        validate_callback_url(str(request.callback_url))

    start_job(
        db=_session,
        request=request,
        project_id=project_id,
        organization_id=organization_id,
    )

    return APIResponse.success_response(
        data=Message(
            message=f"Your response is being generated and will be delivered via callback."
        ),
    )
