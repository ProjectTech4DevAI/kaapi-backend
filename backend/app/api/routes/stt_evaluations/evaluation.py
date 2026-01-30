"""STT evaluation run API routes."""

import logging

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.celery.tasks.stt_evaluation import process_stt_evaluation
from app.crud.stt_evaluations import (
    create_stt_run,
    get_stt_dataset_by_id,
    get_stt_run_by_id,
    list_stt_runs,
    get_sample_count_for_dataset,
)
from app.crud.stt_evaluations.result import get_results_by_run_id
from app.models.stt_evaluation import (
    STTEvaluationRunCreate,
    STTEvaluationRunPublic,
    STTEvaluationRunWithResults,
)
from app.utils import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/runs",
    response_model=APIResponse[STTEvaluationRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Start STT evaluation",
    description="""
Start an STT evaluation run on a dataset.

The evaluation will:
1. Process each audio sample through the specified providers
2. Generate transcriptions using Gemini Batch API
3. Store results for human review

**Supported providers:** gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash
""",
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
        f"providers: {run_create.providers}"
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

    # Get sample count for total items
    sample_count = get_sample_count_for_dataset(
        session=_session, dataset_id=run_create.dataset_id
    )

    if sample_count == 0:
        raise HTTPException(status_code=400, detail="Dataset has no samples")

    # Create run record
    run = create_stt_run(
        session=_session,
        run_name=run_create.run_name,
        dataset_id=run_create.dataset_id,
        dataset_name=dataset.name,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        providers=run_create.providers,
        language=run_create.language or dataset.language,
        total_items=sample_count * len(run_create.providers),
    )

    # Enqueue Celery task
    trace_id = correlation_id.get() or ""

    process_stt_evaluation.apply_async(
        kwargs={
            "evaluation_run_id": run.id,
            "org_id": auth_context.organization_.id,
            "project_id": auth_context.project_.id,
            "trace_id": trace_id,
        },
    )

    logger.info(
        f"[start_stt_evaluation] STT evaluation queued | "
        f"run_id: {run.id}, task queued"
    )

    return APIResponse.success_response(
        data=STTEvaluationRunPublic(
            id=run.id,
            run_name=run.run_name,
            dataset_name=run.dataset_name,
            type=run.type,
            language=run.language,
            providers=run.providers,
            dataset_id=run.dataset_id,
            status=run.status,
            total_items=run.total_items,
            processed_samples=run.processed_samples,
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
    description="List all STT evaluation runs for the current project.",
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
    description="Get an STT evaluation run with its results.",
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
            language=run.language,
            providers=run.providers,
            dataset_id=run.dataset_id,
            status=run.status,
            total_items=run.total_items,
            processed_samples=run.processed_samples,
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
