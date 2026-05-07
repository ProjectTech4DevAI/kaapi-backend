"""Parent-assessment endpoints"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from sqlmodel import Session

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.assessment import (
    build_run_stats,
    compute_run_counts,
    derive_aggregate_error,
    get_assessment_by_id,
    get_assessment_runs_for_assessment,
    list_assessments as list_assessments_crud,
)
from app.models.assessment import (
    Assessment,
    AssessmentPublic,
    AssessmentResponse,
)
from app.models.evaluation import EvaluationDataset
from app.services.assessment.service import retry_assessment as retry_assessment_service
from app.services.assessment.utils import build_assessment_results_response
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_assessment_public(
    session: Session,
    assessment: Assessment,
) -> AssessmentPublic:
    """Build AssessmentPublic with derived counts and run_stats."""
    runs = get_assessment_runs_for_assessment(
        session=session, assessment_id=assessment.id
    )
    counts = compute_run_counts(runs)
    dataset = session.get(EvaluationDataset, assessment.dataset_id)
    return AssessmentPublic(
        id=assessment.id,
        experiment_name=assessment.experiment_name,
        dataset_id=assessment.dataset_id,
        dataset_name=dataset.name if dataset else None,
        status=assessment.status,
        counts=counts,
        run_stats=build_run_stats(runs),
        error_message=derive_aggregate_error(counts),
        organization_id=assessment.organization_id,
        project_id=assessment.project_id,
        inserted_at=assessment.inserted_at,
        updated_at=assessment.updated_at,
    )


@router.post(
    "/assessments/{assessment_id}/retry",
    summary="Retry Assessment",
    description=load_description("assessment/retry_assessment.md"),
    response_model=APIResponse[AssessmentResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def retry_assessment(
    assessment_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentResponse]:
    """Retry a parent assessment using the same dataset/config inputs."""
    assessment = get_assessment_by_id(
        session=session,
        assessment_id=assessment_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    result = retry_assessment_service(
        session=session,
        assessment=assessment,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    return APIResponse.success_response(data=result)


@router.get(
    "/assessments",
    summary="List Assessments Parent details",
    description=load_description("assessment/list_assessments.md"),
    response_model=APIResponse[list[AssessmentPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_assessments(
    session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[AssessmentPublic]]:
    """List assessments."""
    assessments = list_assessments_crud(
        session=session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        limit=limit,
        offset=offset,
    )

    return APIResponse.success_response(
        data=[
            _build_assessment_public(session, assessment) for assessment in assessments
        ]
    )


@router.get(
    "/assessments/{assessment_id}",
    summary="Get Parent Assessment Information",
    description=load_description("assessment/get_assessment.md"),
    response_model=APIResponse[AssessmentPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_assessment(
    assessment_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentPublic]:
    """Get a specific assessment."""
    assessment = get_assessment_by_id(
        session=session,
        assessment_id=assessment_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(
        data=_build_assessment_public(session, assessment)
    )


@router.get(
    "/assessments/{assessment_id}/results",
    summary="Export Assessment Results",
    description=load_description("assessment/export_assessment_results.md"),
    response_model=APIResponse[list[dict[str, Any]]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def export_assessment_results(
    assessment_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
    export_format: Literal["json", "csv", "xlsx"] = Query(default="json"),
) -> APIResponse[list[dict[str, Any]]] | StreamingResponse:
    """Return child-run results. For CSV/XLSX with multiple runs, returns a ZIP."""
    assessment = get_assessment_by_id(
        session=session,
        assessment_id=assessment_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    runs = get_assessment_runs_for_assessment(
        session=session, assessment_id=assessment_id
    )

    return build_assessment_results_response(
        session=session,
        assessment=assessment,
        runs=runs,
        export_format=export_format,
    )
