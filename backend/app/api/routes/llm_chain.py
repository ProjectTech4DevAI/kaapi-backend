import logging

from fastapi import APIRouter, Depends

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import LLMChainRequest, Message
from app.services.llm.chain_executor import start_chain_job
from app.utils import APIResponse, validate_callback_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["llm"])


@router.post(
    "/llm/chain",
    response_model=APIResponse[Message],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def llm_chain(
    _current_user: AuthContextDep, _session: SessionDep, request: LLMChainRequest
):
    project_id = _current_user.project_.id
    organization_id = _current_user.organization_.id

    if request.callback_url:
        validate_callback_url(str(request.callback_url))

    start_chain_job(
        db=_session,
        request=request,
        project_id=project_id,
        organization_id=organization_id,
    )

    return APIResponse.success_response(
        data=Message(
            message="Chain execution started. Results will be delivered via callback."
        )
    )
