import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import SessionDep
from app.api.permissions import Permission, require_permission
from app.crud import onboard_project
from app.models import OnboardingRequest, OnboardingResponse, User
from app.models.onboarding import ONBOARDING_MAX_PAYLOAD_BYTES, OnboardingRequestV2
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Onboarding"])
router_v2 = APIRouter(tags=["Onboarding v2"])


@router.post(
    "/onboard",
    response_model=APIResponse[OnboardingResponse],
    status_code=201,
    description=load_description("onboarding/onboarding.md"),
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    deprecated=True,
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


def enforce_onboarding_payload_limit(request: Request) -> None:
    """Reject oversized bodies as a dependency so 413 wins over body-validation 422.

    Chunked requests carry no Content-Length and are let through rather than
    buffered here; the proxy enforces the ceiling for those.
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > ONBOARDING_MAX_PAYLOAD_BYTES:
        logger.warning(
            f"[enforce_onboarding_payload_limit] Payload too large | content_length: {content_length}, "
            f"limit: {ONBOARDING_MAX_PAYLOAD_BYTES}"
        )
        raise HTTPException(
            status_code=413,
            detail=f"Onboarding payload exceeds the {ONBOARDING_MAX_PAYLOAD_BYTES} byte limit.",
        )


@router_v2.post(
    "/onboard",
    response_model=APIResponse[OnboardingResponse],
    status_code=201,
    description=load_description("onboarding/onboarding.md"),
    dependencies=[
        Depends(require_permission(Permission.SUPERUSER)),
        Depends(enforce_onboarding_payload_limit),
    ],
)
def onboard_project_route_v2(
    onboard_in: OnboardingRequestV2,
    session: SessionDep,
) -> APIResponse[OnboardingResponse]:
    response = onboard_project(session=session, onboard_in=onboard_in)
    logger.info(
        f"[onboard_project_route_v2] Onboarded project | organization_id: {response.organization_id}, "
        f"project_id: {response.project_id}"
    )

    metadata = None
    if onboard_in.credentials:
        metadata = {"note": ("Given credential(s) have been saved for this project.")}

    return APIResponse.success_response(data=response, metadata=metadata)
