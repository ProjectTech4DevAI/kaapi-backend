"""Duplicate-detection routes: submit a check, read its status and result."""

from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models.assessment.duplicate_detection import (
    DuplicateDetectionCreate,
    DuplicateDetectionPublic,
)
from app.services.assessment.duplicate_detection import service
from app.utils import APIResponse, load_description

router = APIRouter(
    prefix="/assessments/duplicates",
    tags=["Assessment (API)"],
)


@router.post(
    "",
    status_code=202,
    description=load_description("assessment/duplicate_detection_create.md"),
    response_model=APIResponse[DuplicateDetectionPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_duplicate_detection(
    request: DuplicateDetectionCreate,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[DuplicateDetectionPublic]:
    job = service.submit(
        session=session,
        request=request,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=job)


@router.get(
    "/{assessment_id}",
    description=load_description("assessment/duplicate_detection_get.md"),
    response_model=APIResponse[DuplicateDetectionPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_duplicate_detection(
    assessment_id: UUID,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[DuplicateDetectionPublic]:
    job = service.get_status(
        session=session,
        assessment_id=assessment_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=job)
