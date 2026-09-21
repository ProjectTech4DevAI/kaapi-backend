"""Assessment API-client route.

Run mode is inferred from the input shape:
    - RESPONSE: single object input (`input` key) — deferred, returns 501.
    - BATCH: submission-list input (`data` key) — N items over one provider batch series.

Results are delivered to the request's `callback_url` when one is given, and are always
readable from `GET /assessments/{assessment_id}`.
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.assessment import api as api_crud
from app.crud.assessment.core import get_assessment_by_id
from app.models.assessment import (
    AssessmentCreate,
    AssessmentDetailResponse,
    AssessmentMethod,
    AssessmentSubmitResponse,
    AssessmentSummary,
    BatchInput,
)
from app.services.assessment.api import results, submission
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/assessments",
    tags=["Assessment (API)"],
)

_RESPONSE_NOT_WIRED = "RESPONSE-mode assessment is not wired yet."


@router.post(
    "",
    description=load_description("assessment/create.md"),
    response_model=APIResponse[AssessmentSubmitResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_assessment(
    request: AssessmentCreate,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentSubmitResponse]:
    """Submit an assessment; the method is inferred from the input shape. The result is
    pushed to `callback_url` when one is given, and is polled from GET otherwise."""
    if not isinstance(request.input, BatchInput):
        raise HTTPException(status_code=501, detail=_RESPONSE_NOT_WIRED)

    result = submission.submit(
        session=session,
        request=request,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    return APIResponse.success_response(data=result)


@router.get(
    "",
    description=load_description("assessment/list_assessments.md"),
    response_model=APIResponse[list[AssessmentSummary]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_assessments(
    session: SessionDep,
    auth_context: AuthContextDep,
    config_id: Annotated[
        UUID | None,
        Query(description="Only runs pinned to this config; omit for every run"),
    ] = None,
    version: Annotated[
        int | None,
        Query(ge=1, description="Config version; omit for every version of config_id"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[list[AssessmentSummary]]:
    """BATCH assessments newest-first, optionally narrowed to one config or version."""
    rows = api_crud.list_assessments_with_execution(
        session=session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        config_id=config_id,
        config_version=version,
        limit=limit,
        offset=offset,
    )
    return APIResponse.success_response(
        data=[
            results.build_summary(assessment, execution, submission_name)
            for assessment, execution, submission_name in rows
        ]
    )


@router.get(
    "/{assessment_id}",
    description=load_description("assessment/get_assessment_detail.md"),
    response_model=APIResponse[AssessmentDetailResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_assessment_detail(
    assessment_id: UUID,
    session: SessionDep,
    auth_context: AuthContextDep,
    include_input: Annotated[
        bool,
        Query(description="Echo each row's submitted columns (one extra storage read)"),
    ] = False,
) -> APIResponse[AssessmentDetailResponse]:
    """Status plus every row produced so far; safe to poll while the run is in flight."""
    assessment = get_assessment_by_id(
        session=session,
        assessment_id=assessment_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    if assessment.method != AssessmentMethod.BATCH:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Assessment {assessment_id} is a {assessment.method} assessment; "
                f"this endpoint serves BATCH only."
            ),
        )

    return APIResponse.success_response(
        data=results.build_detail(
            session=session, assessment=assessment, include_input=include_input
        )
    )
