"""TTS evaluation run API routes."""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.tts_evaluations import (
    create_tts_run,
    create_tts_results,
    get_results_by_run_id,
    get_tts_dataset_by_id,
    get_tts_run_by_id,
    list_tts_runs,
    start_tts_evaluation_batch,
    update_tts_run,
)
from app.models.tts_evaluation import (
    TTSEvaluationRunCreate,
    TTSEvaluationRunPublic,
    TTSEvaluationRunWithResults,
    TTSSampleCreate,
)
from app.services.tts_evaluations.constants import (
    DEFAULT_STYLE_PROMPT,
    DEFAULT_VOICE_NAME,
)
from app.services.tts_evaluations.dataset import parse_tts_samples_from_csv
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
    _session: SessionDep,
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

    # Get sample texts from the dataset CSV
    sample_texts = _get_sample_texts_from_dataset(
        _session, dataset, auth_context.project_.id
    )

    if not sample_texts:
        raise HTTPException(
            status_code=400, detail="Could not load samples from dataset"
        )

    language_id = dataset.language_id

    # Create run record
    run = create_tts_run(
        session=_session,
        run_name=run_create.run_name,
        dataset_id=run_create.dataset_id,
        dataset_name=dataset.name,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        models=run_create.models,
        language_id=language_id,
        total_items=len(sample_texts) * len(run_create.models),
    )

    # Create result records for each sample text and model
    results = create_tts_results(
        session=_session,
        sample_texts=sample_texts,
        evaluation_run_id=run.id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        models=run_create.models,
    )

    try:
        batch_result = start_tts_evaluation_batch(
            session=_session,
            run=run,
            results=results,
            org_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
        )
        logger.info(
            f"[start_tts_evaluation] TTS evaluation batch submitted | "
            f"run_id: {run.id}, batch_jobs: {list(batch_result.get('batch_jobs', {}).keys())}"
        )
    except Exception as e:
        logger.error(
            f"[start_tts_evaluation] Batch submission failed | "
            f"run_id: {run.id}, error: {str(e)}"
        )
        update_tts_run(
            session=_session,
            run_id=run.id,
            status="failed",
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Batch submission failed: {e}")

    # Refresh run to get updated status
    run = get_tts_run_by_id(
        session=_session,
        run_id=run.id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(
        data=TTSEvaluationRunPublic(
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
            run_metadata={
                "voice_name": DEFAULT_VOICE_NAME,
                "style_prompt": DEFAULT_STYLE_PROMPT,
            },
            organization_id=run.organization_id,
            project_id=run.project_id,
            inserted_at=run.inserted_at,
            updated_at=run.updated_at,
        )
    )


def _get_sample_texts_from_dataset(
    session: "SessionDep",
    dataset: "EvaluationDataset",
    project_id: int,
) -> list[str]:
    """Extract sample texts from a TTS dataset's CSV in S3.

    Args:
        session: Database session
        dataset: The evaluation dataset record
        project_id: Project ID

    Returns:
        List of text strings
    """
    if not dataset.object_store_url:
        logger.warning(
            f"[_get_sample_texts_from_dataset] No object_store_url | "
            f"dataset_id={dataset.id}"
        )
        return []

    try:
        from app.core.cloud.storage import get_cloud_storage

        storage = get_cloud_storage(session=session, project_id=project_id)
        csv_bytes = storage.stream(dataset.object_store_url).read()
        samples = parse_tts_samples_from_csv(csv_bytes)
        return [s["text"] for s in samples]
    except Exception as e:
        logger.error(
            f"[_get_sample_texts_from_dataset] Failed to load CSV | "
            f"dataset_id={dataset.id}, error={str(e)}"
        )
        return []


@router.get(
    "/runs",
    response_model=APIResponse[list[TTSEvaluationRunPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="List TTS evaluation runs",
    description=load_description("tts_evaluation/list_runs.md"),
)
def list_tts_evaluation_runs(
    _session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int | None = Query(None, description="Filter by dataset ID"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> APIResponse[list[TTSEvaluationRunPublic]]:
    """List TTS evaluation runs."""
    runs, total = list_tts_runs(
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
    response_model=APIResponse[TTSEvaluationRunWithResults],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Get TTS evaluation run",
    description=load_description("tts_evaluation/get_run.md"),
)
def get_tts_evaluation_run(
    _session: SessionDep,
    auth_context: AuthContextDep,
    run_id: int,
    include_results: bool = Query(True, description="Include results in response"),
) -> APIResponse[TTSEvaluationRunWithResults]:
    """Get a TTS evaluation run with results."""
    run = get_tts_run_by_id(
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
        )

    return APIResponse.success_response(
        data=TTSEvaluationRunWithResults(
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
