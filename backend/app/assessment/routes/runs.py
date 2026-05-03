"""Assessment run endpoints — one row per config-run inside a parent assessment."""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.assessment.crud import (
    get_assessment_by_id,
    get_assessment_run_by_id as get_run_by_id,
    list_assessment_runs as list_runs,
)
from app.assessment.models import (
    Assessment,
    AssessmentCreate,
    AssessmentResponse,
    AssessmentRun,
    AssessmentRunPublic,
)
from app.assessment.service import (
    retry_assessment_run as retry_run,
    start_assessment,
)
from app.assessment.utils import (
    build_export_response,
    build_json_export_rows,
    load_export_rows_for_run,
    sort_export_rows,
)
from app.models.evaluation import EvaluationDataset
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_run_public(
    session: SessionDep,
    run: AssessmentRun,
) -> AssessmentRunPublic:
    """Build AssessmentRunPublic with parent-derived experiment/dataset info."""
    parent = session.get(Assessment, run.assessment_id)
    dataset = session.get(EvaluationDataset, parent.dataset_id) if parent else None
    return AssessmentRunPublic(
        id=run.id,
        assessment_id=run.assessment_id,
        experiment_name=parent.experiment_name if parent else None,
        dataset_id=parent.dataset_id if parent else None,
        dataset_name=dataset.name if dataset else None,
        config_id=run.config_id,
        config_version=run.config_version,
        status=run.status,
        total_items=run.total_items,
        error_message=run.error_message,
        input=run.input,
        inserted_at=run.inserted_at,
        updated_at=run.updated_at,
    )


@router.post(
    "/runs",
    description=load_description("assessment/create_run.md"),
    response_model=APIResponse[AssessmentResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_assessment_runs(
    request: AssessmentCreate,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentResponse]:
    """Submit an assessment and create one child run per config."""
    logger.info(
        f"[create_assessment_runs] Assessment run submission | "
        f"experiment={request.experiment_name} | "
        f"dataset_id={request.dataset_id} | "
        f"configs={len(request.configs)}"
    )

    result = start_assessment(
        session=session,
        request=request,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=result)


@router.post(
    "/runs/{run_id}/retry",
    description=load_description("assessment/retry_run.md"),
    response_model=APIResponse[AssessmentResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def retry_assessment_run(
    run_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentResponse]:
    """Retry a single child assessment run using the same inputs."""
    run = get_run_by_id(
        session=session,
        run_id=run_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment run {run_id} not found or not accessible",
        )

    result = retry_run(
        session=session,
        run=run,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    return APIResponse.success_response(data=result)


@router.get(
    "/runs",
    description=load_description("assessment/list_runs.md"),
    response_model=APIResponse[list[AssessmentRunPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_assessment_runs(
    session: SessionDep,
    auth_context: AuthContextDep,
    assessment_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[AssessmentRunPublic]]:
    """List assessment runs."""
    runs = list_runs(
        session=session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        assessment_id=assessment_id,
        limit=limit,
        offset=offset,
    )

    return APIResponse.success_response(
        data=[_build_run_public(session, run) for run in runs]
    )


@router.get(
    "/runs/{run_id}",
    description=load_description("assessment/get_run.md"),
    response_model=APIResponse[AssessmentRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_assessment_run(
    run_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentRunPublic]:
    """Get a specific assessment run."""
    run = get_run_by_id(
        session=session,
        run_id=run_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment run {run_id} not found or not accessible",
        )

    return APIResponse.success_response(data=_build_run_public(session, run))


@router.get(
    "/runs/{run_id}/results",
    description=load_description("assessment/export_run_results.md"),
    response_model=APIResponse[list[dict[str, Any]]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def export_assessment_run_results(
    run_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
    export_format: Literal["json", "csv", "xlsx"] = Query(default="json"),
) -> APIResponse[list[dict[str, Any]]] | StreamingResponse:
    """Return flattened results for a single child assessment run."""
    run = get_run_by_id(
        session=session,
        run_id=run_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment run {run_id} not found or not accessible",
        )

    assessment = get_assessment_by_id(
        session=session,
        assessment_id=run.assessment_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    export_rows = sort_export_rows(
        load_export_rows_for_run(
            session=session,
            run=run,
            assessment=assessment,
        )
    )

    base_label = assessment.experiment_name if assessment else f"run_{run.id}"
    if export_format != "json":
        return build_export_response(
            export_rows=export_rows,
            export_format=export_format,
            base_name=f"{base_label}_run_{run.id}_results",
        )

    return APIResponse.success_response(data=build_json_export_rows(export_rows))
