"""Celery task function for TTS evaluation batch submission."""

import logging

from gevent import Timeout
from celery.exceptions import SoftTimeLimitExceeded
from sqlmodel import Session

from app.core.db import engine
from app.crud.tts_evaluations.batch import start_tts_evaluation_batch
from app.crud.tts_evaluations.dataset import get_tts_dataset_by_id
from app.crud.tts_evaluations.result import create_tts_results
from app.crud.tts_evaluations.run import get_tts_run_by_id, update_tts_run
from app.services.tts_evaluations.dataset import get_sample_texts_from_dataset

logger = logging.getLogger(__name__)


def execute_batch_submission(
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance,
    organization_id: int,
    dataset_id: int,
    models: list[str],
    **kwargs,
) -> dict:
    """Execute TTS evaluation batch submission in a Celery worker.

    Handles result record creation, JSONL creation, Gemini file upload,
    and batch job creation.

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

    logger.info(
        f"[execute_batch_submission] Starting | "
        f"run_id: {run_id}, project_id: {project_id}, "
        f"celery_task_id: {task_id}"
    )

    with Session(engine) as session:
        try:
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

            sample_texts = get_sample_texts_from_dataset(session, dataset, project_id)

            if not sample_texts:
                logger.warning(
                    f"[execute_batch_submission] No samples found | "
                    f"run_id: {run_id}, dataset_id: {dataset_id}"
                )
                update_tts_run(
                    session=session,
                    run_id=run_id,
                    status="failed",
                    error_message="No samples found for dataset",
                )
                return {"success": False, "error": "No samples found"}

            # Create result records for each sample text and model
            results = create_tts_results(
                session=session,
                sample_texts=sample_texts,
                evaluation_run_id=run.id,
                org_id=organization_id,
                project_id=project_id,
                models=models,
            )

            batch_result = start_tts_evaluation_batch(
                session=session,
                run=run,
                results=results,
                org_id=organization_id,
                project_id=project_id,
            )

            logger.info(
                f"[execute_batch_submission] Batch submitted | "
                f"run_id: {run_id}, "
                f"batch_jobs: {list(batch_result.get('batch_jobs', {}).keys())}"
            )

            return batch_result

        except (Timeout, SoftTimeLimitExceeded) as err:
            timeout_err = TimeoutError("Task exceeded soft time limit")
            logger.warning(
                f"[execute_batch_submission] TTS batch submission timed out | run_id={run_id}"
            )
            update_tts_run(
                session=session,
                run_id=run_id,
                status="failed",
                error_message=str(timeout_err),
            )
            raise

        except Exception as e:
            logger.error(
                f"[execute_batch_submission] Batch submission failed | "
                f"run_id: {run_id}, error: {str(e)}",
                exc_info=True,
            )
            update_tts_run(
                session=session,
                run_id=run_id,
                status="failed",
                error_message=str(e),
            )
            return {"success": False, "error": str(e)}
