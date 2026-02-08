"""STT evaluation run API routes."""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.stt_evaluations import (
    create_stt_run,
    create_stt_results,
    get_results_by_run_id,
    get_samples_by_dataset_id,
    get_stt_dataset_by_id,
    get_stt_run_by_id,
    list_stt_runs,
    start_stt_evaluation_batch,
    update_stt_run,
)
from app.models.stt_evaluation import (
    STTEvaluationRunCreate,
    STTEvaluationRunPublic,
    STTEvaluationRunWithResults,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/runs",
    response_model=APIResponse[STTEvaluationRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Start STT evaluation",
    description=load_description("stt_evaluation/start_evaluation.md"),
)
def start_stt_evaluation(
    _session: SessionDep,
    auth_context: AuthContextDep,
    run_create: STTEvaluationRunCreate = Body(...),
) -> APIResponse[STTEvaluationRunPublic]:
    """Start an STT evaluation run."""
    logger.info(
        f"[start_stt_evaluation] Starting STT evaluation | "
        f"run_name: {run_create.run_name}, dataset_id: {run_create.dataset_id}, "
        f"models: {run_create.models}"
    )

    # Validate dataset exists
    dataset = get_stt_dataset_by_id(
        session=_session,
        dataset_id=run_create.dataset_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    sample_count = (dataset.dataset_metadata or {}).get("sample_count", 0)

    if sample_count == 0:
        raise HTTPException(status_code=400, detail="Dataset has no samples")

    # Use language_id from the dataset
    language_id = dataset.language_id

    # Create run record
    run = create_stt_run(
        session=_session,
        run_name=run_create.run_name,
        dataset_id=run_create.dataset_id,
        dataset_name=dataset.name,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        models=run_create.models,
        language_id=language_id,
        total_items=sample_count * len(run_create.models),
    )

    # Get samples for the dataset
    samples = get_samples_by_dataset_id(
        session=_session,
        dataset_id=run_create.dataset_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    # Create result records for each sample and model
    create_stt_results(
        session=_session,
        samples=samples,
        evaluation_run_id=run.id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        models=run_create.models,
    )

    try:
        batch_result = start_stt_evaluation_batch(
            session=_session,
            run=run,
            samples=samples,
            org_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
        )
        logger.info(
            f"[start_stt_evaluation] STT evaluation batch submitted | "
            f"run_id: {run.id}, batch_jobs: {list(batch_result.get('batch_jobs', {}).keys())}"
        )
    except Exception as e:
        logger.error(
            f"[start_stt_evaluation] Batch submission failed | "
            f"run_id: {run.id}, error: {str(e)}"
        )
        update_stt_run(
            session=_session,
            run_id=run.id,
            status="failed",
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Batch submission failed: {e}")

    # Refresh run to get updated status
    run = get_stt_run_by_id(
        session=_session,
        run_id=run.id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(
        data=STTEvaluationRunPublic(
            id=run.id,
            run_name=run.run_name,
            dataset_name=run.dataset_name,
            type=run.type,
            language_id=run.language_id,
            models=run.providers,
            dataset_id=run.dataset_id,
            status=run.status,
            total_items=run.total_items,
            score=run.score,
            error_message=run.error_message,
            organization_id=run.organization_id,
            project_id=run.project_id,
            inserted_at=run.inserted_at,
            updated_at=run.updated_at,
        )
    )


@router.get(
    "/runs",
    response_model=APIResponse[list[STTEvaluationRunPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="List STT evaluation runs",
    description=load_description("stt_evaluation/list_runs.md"),
)
def list_stt_evaluation_runs(
    _session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int | None = Query(None, description="Filter by dataset ID"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> APIResponse[list[STTEvaluationRunPublic]]:
    """List STT evaluation runs."""
    runs, total = list_stt_runs(
        session=_session,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        dataset_id=dataset_id,
        status=status,
        limit=limit,
        offset=offset,
    )

    return APIResponse.success_response(
        data=runs,
        metadata={"total": total, "limit": limit, "offset": offset},
    )


@router.get(
    "/runs/{run_id}",
    response_model=APIResponse[STTEvaluationRunWithResults],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Get STT evaluation run",
    description=load_description("stt_evaluation/get_run.md"),
)
def get_stt_evaluation_run(
    _session: SessionDep,
    auth_context: AuthContextDep,
    run_id: int,
    include_results: bool = Query(True, description="Include results in response"),
    result_limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
    result_offset: int = Query(0, ge=0, description="Result offset"),
    provider: str | None = Query(None, description="Filter results by provider"),
    status: str | None = Query(None, description="Filter results by status"),
) -> APIResponse[STTEvaluationRunWithResults]:
    """Get an STT evaluation run with results."""
    run = get_stt_run_by_id(
        session=_session,
        run_id=run_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    results = []
    results_total = 0

    if include_results:
        results, results_total = get_results_by_run_id(
            session=_session,
            run_id=run_id,
            org_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            provider=provider,
            status=status,
            limit=result_limit,
            offset=result_offset,
        )

    return APIResponse.success_response(
        data=STTEvaluationRunWithResults(
            id=run.id,
            run_name=run.run_name,
            dataset_name=run.dataset_name,
            type=run.type,
            language_id=run.language_id,
            models=run.providers,
            dataset_id=run.dataset_id,
            status=run.status,
            total_items=run.total_items,
            score=run.score,
            error_message=run.error_message,
            organization_id=run.organization_id,
            project_id=run.project_id,
            inserted_at=run.inserted_at,
            updated_at=run.updated_at,
            results=results,
            results_total=results_total,
        ),
        metadata={"results_total": results_total},
    )
