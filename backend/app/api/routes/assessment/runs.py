"""Assessment run endpoints — one row per config-run inside a parent assessment (LEGACY RUN pipeline).

Serves dataset-based RUN assessments only. The new API-client BATCH path
(`api.py`) delivers results by webhook and never surfaces here.
"""

import logging
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.assessment import (
    get_assessment_by_id,
    update_run_post_processing_config,
)
from app.crud.assessment import (
    get_assessment_run_by_id as get_run_by_id,
)
from app.crud.assessment import (
    list_assessment_runs as list_runs,
)
from app.models.assessment import (
    Assessment,
    AssessmentRun,
    AssessmentRunCreate,
    AssessmentRunPublic,
    AssessmentRunResponse,
    RunExecution,
)
from app.models.evaluation import EvaluationDataset
from app.services.assessment.service import (
    resume_assessment_run as resume_run,
)
from app.services.assessment.service import (
    retry_assessment_run as retry_run,
)
from app.services.assessment.service import (
    start_assessment,
)
from app.services.assessment.utils import (
    build_export_response,
    build_json_export_rows,
    load_export_rows_for_run,
    sort_export_rows,
)
from app.services.assessment.utils.post_processing import apply_post_processing
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_run_public(
    session: SessionDep,
    run: AssessmentRun,
) -> AssessmentRunPublic:
    """Build AssessmentRunPublic with parent-derived experiment/dataset info."""
    parent = session.get(Assessment, run.assessment_id)
    if parent is None:
        logger.warning(
            "[_build_run_public] Parent assessment %s not found for run %s",
            run.assessment_id,
            run.id,
        )
    dataset = session.get(EvaluationDataset, parent.dataset_id) if parent else None
    # Runtime state was folded into the `execution` JSONB bag (migration 078); the frozen
    # request input lives on the parent, not the run. Legacy list/get are scoped to
    # method=RUN, but guard defensively so a non-RUN-shaped bag can never 500 the list.
    try:
        bag = RunExecution.model_validate(run.execution or {})
    except ValidationError:
        logger.warning(
            "[_build_run_public] Non-RUN execution bag for run %s; returning empty bag",
            run.id,
        )
        bag = RunExecution()
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
        input=parent.input if parent else None,
        prefilter_total_rows=bag.prefilter_total_rows,
        prefilter_total_passed=bag.prefilter_total_passed,
        prefilter_total_rejected=bag.prefilter_total_rejected,
        stage=bag.stage,
        stage_status=bag.stage_status,
        pipeline=bag.pipeline,
        cost=bag.cost,
        post_processing_config=run.post_processing_config,
        inserted_at=run.inserted_at,
        updated_at=run.updated_at,
    )


@router.post(
    "/runs",
    description=load_description("assessment/create_run.md"),
    response_model=APIResponse[AssessmentRunResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_assessment_runs(
    request: AssessmentRunCreate,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentRunResponse]:
    """Submit an assessment and create one child run per config."""
    logger.info(
        "[create_assessment_runs] Assessment run submission | experiment=%s | dataset_id=%s | configs=%s",
        request.experiment_name,
        request.dataset_id,
        len(request.configs),
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
    response_model=APIResponse[AssessmentRunResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def retry_assessment_run(
    run_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentRunResponse]:
    """Retry a single child assessment run using the same inputs."""
    run = get_run_by_id(
        session=session,
        run_id=run_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    result = retry_run(
        session=session,
        run=run,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    return APIResponse.success_response(data=result)


@router.post(
    "/runs/{run_id}/resume",
    description=load_description("assessment/resume_run.md"),
    response_model=APIResponse[AssessmentRunResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def resume_assessment_run(
    run_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentRunResponse]:
    """Resume a failed child run from its failed stage, reusing completed stages."""
    run = get_run_by_id(
        session=session,
        run_id=run_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    result = resume_run(
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
    assessment_id: UUID | None = Query(default=None),
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

    post_processing_config = run.post_processing_config or None
    base_label = assessment.experiment_name if assessment else f"run_{run.id}"

    if export_format != "json":
        return build_export_response(
            export_rows=export_rows,
            export_format=export_format,
            base_name=f"{base_label}_run_{run.id}_results",
            post_processing_config=post_processing_config,
        )

    rows = build_json_export_rows(export_rows)
    rows = apply_post_processing(rows, post_processing_config)
    return APIResponse.success_response(data=rows)


@router.patch(
    "/runs/{run_id}/post-processing",
    description=load_description("assessment/update_post_processing.md"),
    response_model=APIResponse[AssessmentRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def update_post_processing(
    run_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
    config: dict[str, Any] | None = Body(default=None),
) -> APIResponse[AssessmentRunPublic]:
    """Save post-processing config (computed columns, sort, filter) for a run."""
    run = get_run_by_id(
        session=session,
        run_id=run_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    run = update_run_post_processing_config(session=session, run=run, config=config)

    return APIResponse.success_response(data=_build_run_public(session, run))
