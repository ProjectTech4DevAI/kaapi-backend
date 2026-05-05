"""TTS evaluation run API routes."""

import logging

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.celery.utils import start_tts_batch_submission
from app.core.cloud import get_cloud_storage
from app.crud.tts_evaluations import (
    create_tts_run,
    get_results_by_run_id,
    get_tts_dataset_by_id,
    get_tts_run_by_id,
    list_tts_runs,
    update_tts_run,
)
from app.models.tts_evaluation import (
    TTSEvaluationRunCreate,
    TTSEvaluationRunPublic,
    TTSEvaluationRunWithResults,
)
from app.services.tts_evaluations.constants import (
    DEFAULT_STYLE_PROMPT,
    DEFAULT_VOICE_NAME,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/runs",
    response_model=APIResponse[TTSEvaluationRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Start TTS evaluation",
    description=load_description("tts_evaluation/start_evaluation.md"),
)
def start_tts_evaluation(
    session: SessionDep,
    auth_context: AuthContextDep,
    run_create: TTSEvaluationRunCreate = Body(...),
) -> APIResponse[TTSEvaluationRunPublic]:
    """Start a TTS evaluation run."""
    logger.info(
        f"[start_tts_evaluation] Starting TTS evaluation | "
        f"run_name: {run_create.run_name}, dataset_id: {run_create.dataset_id}, "
        f"models: {run_create.models}"
    )

    # Validate dataset exists
    dataset = get_tts_dataset_by_id(
        session=session,
        dataset_id=run_create.dataset_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    sample_count = (dataset.dataset_metadata or {}).get("sample_count", 0)

    if sample_count == 0:
        raise HTTPException(status_code=400, detail="Dataset has no samples")

    language_id = dataset.language_id

    # Create run record
    run = create_tts_run(
        session=session,
        run_name=run_create.run_name,
        dataset_id=run_create.dataset_id,
        dataset_name=dataset.name,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        models=run_create.models,
        language_id=language_id,
        total_items=sample_count * len(run_create.models),
    )

    # Offload batch submission (result creation, JSONL, Gemini upload) to Celery worker
    trace_id = correlation_id.get() or "N/A"
    try:
        celery_task_id = start_tts_batch_submission(
            project_id=auth_context.project_.id,
            job_id=str(run.id),
            trace_id=trace_id,
            organization_id=auth_context.organization_.id,
            dataset_id=run_create.dataset_id,
            models=run_create.models,
        )
        logger.info(
            f"[start_tts_evaluation] Batch submission queued | "
            f"run_id: {run.id}, celery_task_id: {celery_task_id}"
        )
    except Exception as e:
        logger.error(
            f"[start_tts_evaluation] Failed to queue batch submission | "
            f"run_id: {run.id}, error: {str(e)}"
        )
        update_tts_run(
            session=session,
            run_id=run.id,
            status="failed",
            error_message=f"Failed to queue batch submission: {str(e)}",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue batch submission: {e}",
        )

    return APIResponse.success_response(
        data=TTSEvaluationRunPublic.from_model(
            run,
            run_metadata={
                "voice_name": DEFAULT_VOICE_NAME,
                "style_prompt": DEFAULT_STYLE_PROMPT,
            },
        )
    )


@router.get(
    "/runs",
    response_model=APIResponse[list[TTSEvaluationRunPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="List TTS evaluation runs",
    description=load_description("tts_evaluation/list_runs.md"),
)
def list_tts_evaluation_runs(
    session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int | None = Query(None, description="Filter by dataset ID"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> APIResponse[list[TTSEvaluationRunPublic]]:
    """List TTS evaluation runs."""
    runs, total = list_tts_runs(
        session=session,
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
    response_model=APIResponse[TTSEvaluationRunWithResults],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Get TTS evaluation run",
    description=load_description("tts_evaluation/get_run.md"),
)
def get_tts_evaluation_run(
    session: SessionDep,
    auth_context: AuthContextDep,
    run_id: int,
    include_results: bool = Query(True, description="Include results in response"),
    include_signed_url: bool = Query(
        False, description="Include signed URLs for generated audio files"
    ),
) -> APIResponse[TTSEvaluationRunWithResults]:
    """Get a TTS evaluation run with results."""
    run = get_tts_run_by_id(
        session=session,
        run_id=run_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    results = []
    results_total = 0

    if include_results:
        storage = None
        if include_signed_url:
            storage = get_cloud_storage(
                session=session, project_id=auth_context.project_.id
            )

        results, results_total = get_results_by_run_id(
            session=session,
            run_id=run_id,
            org_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            storage=storage,
        )

    return APIResponse.success_response(
        data=TTSEvaluationRunWithResults.from_model(
            run,
            results=results,
            results_total=results_total,
        ),
        metadata={"results_total": results_total},
    )
