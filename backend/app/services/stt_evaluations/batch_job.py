"""Celery task function for STT evaluation batch submission."""

import logging

from sqlmodel import Session

from app.core.db import engine
from app.crud.stt_evaluations.batch import start_stt_evaluation_batch
from app.crud.stt_evaluations.dataset import get_samples_by_dataset_id
from app.crud.stt_evaluations.run import get_stt_run_by_id, update_stt_run

logger = logging.getLogger(__name__)


def execute_batch_submission(
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance,
    organization_id: int,
    dataset_id: int,
    **kwargs,
) -> dict:
    """Execute STT evaluation batch submission in a Celery worker.

    Handles signed URL generation, JSONL creation, Gemini file upload,
    and batch job creation.

    Args:
        project_id: Project ID
        job_id: Evaluation run ID (as string)
        task_id: Celery task ID
        task_instance: Celery task instance
        organization_id: Organization ID
        dataset_id: Dataset ID

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
        run = get_stt_run_by_id(
            session=session,
            run_id=run_id,
            org_id=organization_id,
            project_id=project_id,
        )

        if not run:
            logger.error(f"[execute_batch_submission] Run not found | run_id: {run_id}")
            return {"success": False, "error": "Run not found"}

        samples = get_samples_by_dataset_id(
            session=session,
            dataset_id=dataset_id,
            org_id=organization_id,
            project_id=project_id,
            limit=run.total_items,
        )

        if not samples:
            logger.error(
                f"[execute_batch_submission] No samples found | "
                f"run_id: {run_id}, dataset_id: {dataset_id}"
            )
            update_stt_run(
                session=session,
                run_id=run_id,
                status="failed",
                error_message="No samples found for dataset",
            )
            return {"success": False, "error": "No samples found"}

        try:
            batch_result = start_stt_evaluation_batch(
                session=session,
                run=run,
                samples=samples,
                org_id=organization_id,
                project_id=project_id,
            )

            logger.info(
                f"[execute_batch_submission] Batch submitted | "
                f"run_id: {run_id}, "
                f"batch_jobs: {list(batch_result.get('batch_jobs', {}).keys())}"
            )

            return batch_result

        except Exception as e:
            logger.error(
                f"[execute_batch_submission] Batch submission failed | "
                f"run_id: {run_id}, error: {str(e)}",
                exc_info=True,
            )
            update_stt_run(
                session=session,
                run_id=run_id,
                status="failed",
                error_message=str(e),
            )
            return {"success": False, "error": str(e)}
