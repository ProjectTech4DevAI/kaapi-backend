"""Celery task function for TTS evaluation submission.

Creates result rows, submits batch-mode models (Gemini) to the provider Batch API
and fans sync-mode models (Sarvam, ElevenLabs) out to chunked Celery tasks.
"""

import logging
from dataclasses import dataclass
from typing import Any

from asgi_correlation_id import correlation_id
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session

from app.celery.utils import start_tts_sync_chunk
from app.core.cloud.storage import CloudStorage, get_cloud_storage
from app.core.db import engine
from app.crud.tts_evaluations.batch import (
    TTSBatchSubmissionError,
    start_tts_evaluation_batch,
)
from app.crud.tts_evaluations.dataset import get_tts_dataset_by_id
from app.crud.tts_evaluations.result import (
    bulk_update_pending_tts_results,
    create_tts_results,
    get_pending_results_for_run,
)
from app.crud.tts_evaluations.run import (
    finalize_tts_run_status,
    get_tts_run_by_id,
    update_tts_run,
)
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult, TTSResultUpdate
from app.services.tts_evaluations.constants import (
    KAAPI_TAG,
    TTS_SYNC_CHUNK_SIZE,
    split_models_by_execution_mode,
)
from app.services.tts_evaluations.dataset import load_sample_texts
from app.services.tts_evaluations.language import resolve_dataset_tts_language_code

logger = logging.getLogger(__name__)

DEFAULT_TRACE_ID = "N/A"


@dataclass(frozen=True)
class _SubmissionContext:
    dataset_id: int
    object_store_url: str | None
    language_code: str
    storage: CloudStorage | None


def execute_batch_submission(
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance: Task,
    organization_id: int,
    dataset_id: int,
    models: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute TTS evaluation submission in a Celery worker.

    Handles result record creation, Gemini batch submission for batch-mode
    models and chunked Celery fan-out for sync-mode models.

    Args:
        project_id: Project ID
        job_id: Evaluation run ID (as string)
        task_id: Celery task ID
        task_instance: Celery task instance
        organization_id: Organization ID
        dataset_id: Dataset ID
        models: List of TTS model names to evaluate

    Returns:
        dict: Result summary with batch job and sync chunk info
    """
    run_id = int(job_id)

    logger.info(
        f"[execute_batch_submission] Starting | "
        f"run_id: {run_id}, project_id: {project_id}, "
        f"celery_task_id: {task_id}"
    )

    try:
        batch_models, sync_models = split_models_by_execution_mode(models)

        context = _load_submission_context(
            run_id=run_id,
            organization_id=organization_id,
            project_id=project_id,
            dataset_id=dataset_id,
        )
        if isinstance(context, dict):
            return context

        # S3 read happens with no session open.
        sample_texts: list[str] = []
        if context.object_store_url and context.storage is not None:
            sample_texts = load_sample_texts(
                storage=context.storage,
                object_store_url=context.object_store_url,
                dataset_id=context.dataset_id,
            )
        else:
            logger.warning(
                f"[execute_batch_submission] No object_store_url | "
                f"run_id: {run_id}, dataset_id: {dataset_id}"
            )

        if not sample_texts:
            logger.warning(
                f"[execute_batch_submission] No samples found | "
                f"run_id: {run_id}, dataset_id: {dataset_id}"
            )
            _mark_run_failed(
                run_id=run_id, error_message="No samples found for dataset"
            )
            return {"success": False, "error": "No samples found"}

        batch_result: dict[str, Any] = {}
        sync_result_ids: dict[str, list[int]] = {}
        result_count = 0

        with Session(engine) as session:
            run = get_tts_run_by_id(
                session=session,
                run_id=run_id,
                org_id=organization_id,
                project_id=project_id,
            )
            if not run:
                logger.warning(
                    f"[execute_batch_submission] Run not found | run_id: {run_id}"
                )
                return {"success": False, "error": "Run not found"}

            create_tts_results(
                session=session,
                sample_texts=sample_texts,
                evaluation_run_id=run_id,
                org_id=organization_id,
                project_id=project_id,
                models=models,
            )
            # One SELECT reloads every row create_tts_results' commit expired,
            # instead of one lazy reload per row below.
            results = get_pending_results_for_run(session=session, run_id=run_id)
            result_count = len(results)

            batch_results: list[TTSResult] = []
            for result in results:
                if result.provider in batch_models:
                    batch_results.append(result)
                elif result.provider in sync_models:
                    sync_result_ids.setdefault(result.provider, []).append(result.id)

            if batch_models:
                try:
                    batch_result = start_tts_evaluation_batch(
                        session=session,
                        run=run,
                        results=batch_results,
                        models=batch_models,
                        org_id=organization_id,
                        project_id=project_id,
                    )
                except TTSBatchSubmissionError as e:
                    # Batch rows are already FAILED; sync models can still finish the run.
                    if not sync_models:
                        raise
                    logger.warning(
                        f"[execute_batch_submission] Batch models failed, "
                        f"continuing with sync models | run_id: {run_id}, "
                        f"error: {str(e)}"
                    )

            if sync_models:
                # Must precede the enqueue: a fast chunk finalizing the run would
                # otherwise be overwritten back to "processing" and strand it.
                update_tts_run(session=session, run_id=run_id, status="processing")

        sync_chunks: dict[str, list[str]] = {}
        if sync_result_ids:
            sync_chunks = _dispatch_sync_chunks(
                run_id=run_id,
                organization_id=organization_id,
                project_id=project_id,
                result_ids_by_model=sync_result_ids,
                language_code=context.language_code,
            )

        logger.info(
            f"[execute_batch_submission] Submitted | run_id: {run_id}, "
            f"batch_jobs: {list(batch_result.get('batch_jobs', {}).keys())}, "
            f"sync_models: {list(sync_chunks.keys())}"
        )

        return {
            "success": True,
            "run_id": run_id,
            "batch_jobs": batch_result.get("batch_jobs", {}),
            "sync_chunks": sync_chunks,
            "result_count": result_count,
        }

    except (Timeout, SoftTimeLimitExceeded):
        timeout_err = TimeoutError("Task exceeded soft time limit")
        logger.warning(
            f"[execute_batch_submission] TTS batch submission timed out | run_id={run_id}"
        )
        _mark_run_failed(run_id=run_id, error_message=str(timeout_err))
        raise

    except Exception as e:
        logger.error(
            f"[execute_batch_submission] Batch submission failed | "
            f"run_id: {run_id}, error: {str(e)}",
            exc_info=True,
        )
        _mark_run_failed(run_id=run_id, error_message=str(e))
        return {"success": False, "error": str(e)}


def _load_submission_context(
    *,
    run_id: int,
    organization_id: int,
    project_id: int,
    dataset_id: int,
) -> _SubmissionContext | dict[str, Any]:
    """Validate run + dataset and capture what the session-free steps need.

    Returns a failure dict (already persisted on the run) when validation fails.
    """
    with Session(engine) as session:
        run = get_tts_run_by_id(
            session=session,
            run_id=run_id,
            org_id=organization_id,
            project_id=project_id,
        )
        if not run:
            logger.warning(
                f"[_load_submission_context] Run not found | run_id: {run_id}"
            )
            return {"success": False, "error": "Run not found"}

        dataset = get_tts_dataset_by_id(
            session=session,
            dataset_id=dataset_id,
            org_id=organization_id,
            project_id=project_id,
        )
        if not dataset:
            logger.warning(
                f"[_load_submission_context] Dataset not found | "
                f"run_id: {run_id}, dataset_id: {dataset_id}"
            )
            update_tts_run(
                session=session,
                run_id=run_id,
                status="failed",
                error_message="Dataset not found",
            )
            return {"success": False, "error": "Dataset not found"}

        object_store_url = dataset.object_store_url
        language_code = resolve_dataset_tts_language_code(
            session=session, language_id=dataset.language_id
        )
        storage = (
            get_cloud_storage(session=session, project_id=project_id)
            if object_store_url
            else None
        )

    logger.info(
        f"[_load_submission_context] Context loaded | run_id: {run_id}, "
        f"dataset_id: {dataset_id}, language: {language_code}"
    )

    return _SubmissionContext(
        dataset_id=dataset_id,
        object_store_url=object_store_url,
        language_code=language_code,
        storage=storage,
    )


def _dispatch_sync_chunks(
    *,
    run_id: int,
    organization_id: int,
    project_id: int,
    result_ids_by_model: dict[str, list[int]],
    language_code: str,
) -> dict[str, list[str]]:
    """Enqueue one `run_tts_sync_chunk` per (model, slice of result IDs).

    A chunk that can't be enqueued has its results marked FAILED, since no
    other path would ever synthesize them.

    Returns:
        dict[str, list[str]]: Celery task IDs per model
    """
    trace_id = correlation_id.get() or DEFAULT_TRACE_ID
    task_ids_by_model: dict[str, list[str]] = {}
    failed_updates: list[TTSResultUpdate] = []

    for model, result_ids in result_ids_by_model.items():
        task_ids: list[str] = []
        for start in range(0, len(result_ids), TTS_SYNC_CHUNK_SIZE):
            chunk_ids = result_ids[start : start + TTS_SYNC_CHUNK_SIZE]
            try:
                celery_task_id = start_tts_sync_chunk(
                    project_id=project_id,
                    job_id=str(run_id),
                    trace_id=trace_id,
                    organization_id=organization_id,
                    model=model,
                    result_ids=chunk_ids,
                    language_code=language_code,
                )
            except Exception as e:
                logger.error(
                    f"[_dispatch_sync_chunks] Failed to enqueue chunk | "
                    f"run_id: {run_id}, model: {model}, "
                    f"chunk_size: {len(chunk_ids)}, error: {str(e)}",
                    exc_info=True,
                )
                for result_id in chunk_ids:
                    failed_updates.append(
                        TTSResultUpdate(
                            result_id=result_id,
                            status=JobStatus.FAILED,
                            error_message=(
                                f"[{KAAPI_TAG}] Failed to queue synthesis for {model}: "
                                f"{str(e)}. Re-run the evaluation; contact Kaapi "
                                f"if it persists."
                            ),
                        )
                    )
                continue
            task_ids.append(celery_task_id)

        task_ids_by_model[model] = task_ids
        logger.info(
            f"[_dispatch_sync_chunks] Model dispatched | run_id: {run_id}, "
            f"model: {model}, results: {len(result_ids)}, chunks: {len(task_ids)}"
        )

    if failed_updates:
        with Session(engine) as session:
            bulk_update_pending_tts_results(session=session, updates=failed_updates)
            finalize_tts_run_status(session=session, run_id=run_id)

    return task_ids_by_model


def _mark_run_failed(*, run_id: int, error_message: str) -> None:
    with Session(engine) as session:
        update_tts_run(
            session=session,
            run_id=run_id,
            status="failed",
            error_message=error_message,
        )
