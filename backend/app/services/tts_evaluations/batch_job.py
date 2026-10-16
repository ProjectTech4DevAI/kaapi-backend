"""Celery task function for TTS evaluation submission.

Gemini models are submitted to the Gemini Batch API; each Sarvam/ElevenLabs model
gets its own `run_tts_sync_generation` Celery task. The cron sets the final run status.
"""

import logging
from typing import Any

from asgi_correlation_id import correlation_id
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session

from app.celery.utils import start_tts_sync_generation
from app.core.cloud.storage import get_cloud_storage
from app.core.db import engine
from app.crud.tts_evaluations.batch import start_tts_evaluation_batch
from app.crud.tts_evaluations.dataset import get_tts_dataset_by_id
from app.crud.tts_evaluations.result import (
    create_tts_results,
    get_pending_results_for_run,
    update_tts_result,
)
from app.crud.tts_evaluations.run import get_tts_run_by_id, update_tts_run
from app.models import Language
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.tts_evaluations.constants import (
    DEFAULT_TTS_LANGUAGE,
    GEMINI_TTS_MODELS,
    SYNC_TTS_MODELS,
    LOCALE_TO_BCP47,
)
from app.services.tts_evaluations.dataset import get_sample_texts_from_dataset

logger = logging.getLogger(__name__)


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

    Creates result rows, submits the Gemini batch and queues one sync task per
    Sarvam/ElevenLabs model. A model that fails to start has its rows marked
    FAILED; the other models carry on.

    Args:
        project_id: Project ID
        job_id: Evaluation run ID (as string)
        task_id: Celery task ID
        task_instance: Celery task instance
        organization_id: Organization ID
        dataset_id: Dataset ID
        models: List of TTS model names to evaluate

    Returns:
        dict: Result summary with batch job info
    """
    run_id = int(job_id)
    batch_models = [model for model in models if model in GEMINI_TTS_MODELS]
    sync_models = [model for model in models if model in SYNC_TTS_MODELS]

    logger.info(
        f"[execute_batch_submission] Starting | "
        f"run_id: {run_id}, project_id: {project_id}, "
        f"celery_task_id: {task_id}"
    )

    try:
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

            dataset = get_tts_dataset_by_id(
                session=session,
                dataset_id=dataset_id,
                org_id=organization_id,
                project_id=project_id,
            )
            if not dataset:
                logger.warning(
                    f"[execute_batch_submission] Dataset not found | "
                    f"run_id: {run_id}, dataset_id: {dataset_id}"
                )
                update_tts_run(
                    session=session,
                    run_id=run_id,
                    status="failed",
                    error_message="Dataset not found",
                )
                return {"success": False, "error": "Dataset not found"}

            language_code = DEFAULT_TTS_LANGUAGE
            if dataset.language_id is not None:
                language = session.get(Language, dataset.language_id)
                if language:
                    language_code = LOCALE_TO_BCP47.get(
                        language.locale, DEFAULT_TTS_LANGUAGE
                    )

            storage = get_cloud_storage(session=session, project_id=project_id)

        # Read from S3 storage with no db session open
        sample_texts = get_sample_texts_from_dataset(storage, dataset)
        if not sample_texts:
            logger.warning(
                f"[execute_batch_submission] No samples found | "
                f"run_id: {run_id}, dataset_id: {dataset_id}"
            )
            _mark_run_failed(
                run_id=run_id, error_message="No samples found for dataset"
            )
            return {"success": False, "error": "No samples found"}

        with Session(engine) as session:
            results = create_tts_results(
                session=session,
                sample_texts=sample_texts,
                evaluation_run_id=run_id,
                org_id=organization_id,
                project_id=project_id,
                models=models,
            )
            result_count = len(results)
            update_tts_run(session=session, run_id=run_id, status="processing")

        # model -> error; these models' rows are marked FAILED once all models are started.
        failed_models: dict[str, str] = {}

        # Queue sync tasks first: queuing is instant, while Gemini submission
        # (JSONL upload + batch create) takes seconds per model.
        trace_id = correlation_id.get() or "N/A"
        for model in sync_models:
            try:
                start_tts_sync_generation(
                    project_id=project_id,
                    job_id=str(run_id),
                    trace_id=trace_id,
                    organization_id=organization_id,
                    model=model,
                    language_code=language_code,
                )
            except Exception as e:
                logger.error(
                    f"[execute_batch_submission] Failed to queue sync generation | "
                    f"run_id: {run_id}, model: {model}, error: {str(e)}",
                    exc_info=True,
                )
                failed_models[model] = f"Failed to queue synthesis: {str(e)}"

        batch_result: dict[str, Any] = {}
        if batch_models:
            with Session(engine) as session:
                batch_results: list[TTSResult] = []
                for model in batch_models:
                    batch_results.extend(
                        get_pending_results_for_run(
                            session=session, run_id=run_id, provider=model
                        )
                    )
                try:
                    batch_result = start_tts_evaluation_batch(
                        session=session,
                        run=run,
                        results=batch_results,
                        org_id=organization_id,
                        project_id=project_id,
                    )
                except Exception as e:
                    logger.error(
                        f"[execute_batch_submission] Gemini batch submission failed | "
                        f"run_id: {run_id}, error: {str(e)}",
                        exc_info=True,
                    )
                    for model in batch_models:
                        failed_models[model] = f"Batch submission failed: {str(e)}"

        if failed_models:
            with Session(engine) as session:
                for model, error_message in failed_models.items():
                    pending = get_pending_results_for_run(
                        session=session, run_id=run_id, provider=model
                    )
                    for result in pending:
                        update_tts_result(
                            session=session,
                            result_id=result.id,
                            status=JobStatus.FAILED.value,
                            error_message=error_message,
                        )
                session.commit()

        logger.info(
            f"[execute_batch_submission] Submitted | run_id: {run_id}, "
            f"batch_jobs: {list(batch_result.get('batch_jobs', {}).keys())}, "
            f"sync_models: {sync_models}, failed_models: {list(failed_models.keys())}"
        )

        return {
            "success": True,
            "run_id": run_id,
            "batch_jobs": batch_result.get("batch_jobs", {}),
            "result_count": result_count,
        }

    except (Timeout, SoftTimeLimitExceeded):
        logger.warning(
            f"[execute_batch_submission] TTS batch submission timed out | run_id={run_id}"
        )
        _mark_run_failed(run_id=run_id, error_message="Task exceeded soft time limit")
        raise

    except Exception as e:
        logger.error(
            f"[execute_batch_submission] Batch submission failed | "
            f"run_id: {run_id}, error: {str(e)}",
            exc_info=True,
        )
        _mark_run_failed(run_id=run_id, error_message=str(e))
        return {"success": False, "error": str(e)}


def _mark_run_failed(*, run_id: int, error_message: str) -> None:
    with Session(engine) as session:
        update_tts_run(
            session=session,
            run_id=run_id,
            status="failed",
            error_message=error_message,
        )
