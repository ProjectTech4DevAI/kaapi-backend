import logging

from fastapi import APIRouter, Depends

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import LLMChainRequest, LLMChainResponse, Message
from app.services.llm.jobs import start_chain_job
from app.utils import APIResponse, load_description, validate_callback_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["LLM"])
llm_callback_router = APIRouter()


@llm_callback_router.post(
    "{$callback_url}",
    name="llm_chain_callback",
)
def llm_callback_notification(body: APIResponse[LLMChainResponse]):
    """
    Callback endpoint specification for LLM chain completion.

    The callback will receive:
    - On success: APIResponse with success=True and data containing LLMChainResponse
    - On failure: APIResponse with success=False and error message
    - metadata field will always be included if provided in the request
    """
    ...


@router.post(
    "/llm/chain",
    description=load_description("llm/llm_chain.md"),
    response_model=APIResponse[Message],
    callbacks=llm_callback_router.routes,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def llm_chain(
    current_user: AuthContextDep,
    session: SessionDep,
    request: LLMChainRequest,
) -> APIResponse[Message]:
    """
    Endpoint to initiate an LLM chain as a background job.
    """
    project_id = current_user.project_.id
    organization_id = current_user.organization_.id

    if request.callback_url:
        validate_callback_url(str(request.callback_url))

    start_chain_job(
        db=session,
        request=request,
        project_id=project_id,
        organization_id=organization_id,
    )

    return APIResponse.success_response(
        data=Message(
            message="Your response is being generated and will be delivered via callback."
        ),
    )
