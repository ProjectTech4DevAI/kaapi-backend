from fastapi import APIRouter, Depends

from app.api.deps import SessionDep
from app.api.permissions import Permission, require_permission
from app.crud import onboard_project
from app.models import OnboardingRequest, OnboardingResponse, User
from app.utils import APIResponse, load_description

router = APIRouter(tags=["Onboarding"])


@router.post(
    "/onboard",
    response_model=APIResponse[OnboardingResponse],
    status_code=201,
    description=load_description("onboarding/onboarding.md"),
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def onboard_project_route(
    onboard_in: OnboardingRequest,
    session: SessionDep,
):
    response = onboard_project(session=session, onboard_in=onboard_in)

    metadata = None
    if onboard_in.credentials:
        metadata = {"note": ("Given credential(s) have been saved for this project.")}

    return APIResponse.success_response(data=response, metadata=metadata)
