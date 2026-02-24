"""Cron processing functions for TTS evaluations.

This module provides functions that are called periodically to process
pending TTS evaluations - polling batch status and dispatching result
processing to Celery workers.

Follows the same pattern as STT evaluations: single query to fetch all
processing runs, grouped by project_id for credential management.
"""

import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import Integer
from sqlmodel import Session, select

from app.celery.utils import start_low_priority_job
from app.core.batch import (
    BatchJobState,
    GeminiBatchProvider,
    poll_batch_status,
)
from app.crud.tts_evaluations.result import (
    count_results_by_status,
    get_pending_results_for_run,
)
from app.crud.tts_evaluations.run import update_tts_run
from app.models import EvaluationRun
from app.models.batch_job import BatchJob
from app.models.job import JobStatus
from app.models.stt_evaluation import EvaluationType
from app.services.stt_evaluations.gemini import GeminiClient

logger = logging.getLogger(__name__)

# Terminal states that indicate batch processing is complete
TERMINAL_STATES = {
    BatchJobState.SUCCEEDED.value,
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
}

# Function path for Celery task dispatch
_TTS_RESULT_PROCESSING_PATH = (
    "app.services.tts_evaluations.batch_result_processing.execute_tts_result_processing"
)


async def poll_all_pending_tts_evaluations(
    session: Session,
) -> dict[str, Any]:
    """Poll all pending TTS evaluations across all organizations.

    Fetches all TTS evaluation runs with status='processing' in a single query,
    groups them by project_id, and processes each project with its own
    Gemini client.

    Args:
        session: Database session

    Returns:
        Summary dict with total, processed, failed, still_processing counts
    """
    logger.info("[poll_all_pending_tts_evaluations] Starting TTS evaluation polling")

    # Single query to fetch all processing TTS evaluation runs
    statement = select(EvaluationRun).where(
        EvaluationRun.type == EvaluationType.TTS.value,
        EvaluationRun.status == "processing",
        EvaluationRun.batch_job_id.is_not(None),
    )
    pending_runs = session.exec(statement).all()

    if not pending_runs:
        logger.info("[poll_all_pending_tts_evaluations] No pending TTS runs found")
        return {
            "total": 0,
            "processed": 0,
            "failed": 0,
            "still_processing": 0,
            "details": [],
        }

    logger.info(
        f"[poll_all_pending_tts_evaluations] Found {len(pending_runs)} pending TTS runs"
    )

    # Group evaluations by project_id since credentials are per project
    evaluations_by_project: dict[int, list[EvaluationRun]] = defaultdict(list)
    for run in pending_runs:
        evaluations_by_project[run.project_id].append(run)

    # Process each project separately
    all_results: list[dict[str, Any]] = []
    total_processed = 0
    total_failed = 0
    total_still_processing = 0

    for project_id, project_runs in evaluations_by_project.items():
        org_id = project_runs[0].organization_id

        try:
            try:
                gemini_client = GeminiClient.from_credentials(
                    session=session,
                    org_id=org_id,
                    project_id=project_id,
                )
            except Exception as client_err:
                logger.error(
                    f"[poll_all_pending_tts_evaluations] Failed to get Gemini client | "
                    f"org_id={org_id} | project_id={project_id} | error={client_err}"
                )
                for run in project_runs:
                    update_tts_run(
                        session=session,
                        run_id=run.id,
                        status="failed",
                        error_message=f"Gemini client initialization failed: {str(client_err)}",
                    )
                    all_results.append(
                        {
                            "run_id": run.id,
                            "run_name": run.run_name,
                            "type": "tts",
                            "action": "failed",
                            "error": str(client_err),
                        }
                    )
                    total_failed += 1
                continue

            batch_provider = GeminiBatchProvider(client=gemini_client.client)

            for run in project_runs:
                try:
                    result = await poll_tts_run(
                        session=session,
                        run=run,
                        batch_provider=batch_provider,
                        org_id=org_id,
                    )
                    all_results.append(result)

                    if result["action"] in ("completed", "processed", "dispatched"):
                        total_processed += 1
                    elif result["action"] == "failed":
                        total_failed += 1
                    else:
                        total_still_processing += 1

                except Exception as e:
                    logger.error(
                        f"[poll_all_pending_tts_evaluations] Failed to poll TTS run | "
                        f"run_id={run.id} | {e}",
                        exc_info=True,
                    )
                    update_tts_run(
                        session=session,
                        run_id=run.id,
                        status="failed",
                        error_message=f"Polling failed: {str(e)}",
                    )
                    all_results.append(
                        {
                            "run_id": run.id,
                            "run_name": run.run_name,
                            "type": "tts",
                            "action": "failed",
                            "error": str(e),
                        }
                    )
                    total_failed += 1

        except Exception as e:
            logger.error(
                f"[poll_all_pending_tts_evaluations] Failed to process project | "
                f"project_id={project_id} | {e}",
                exc_info=True,
            )
            for run in project_runs:
                update_tts_run(
                    session=session,
                    run_id=run.id,
                    status="failed",
                    error_message=f"Project processing failed: {str(e)}",
                )
                all_results.append(
                    {
                        "run_id": run.id,
                        "run_name": run.run_name,
                        "type": "tts",
                        "action": "failed",
                        "error": f"Project processing failed: {str(e)}",
                    }
                )
                total_failed += 1

    summary = {
        "total": len(pending_runs),
        "processed": total_processed,
        "failed": total_failed,
        "still_processing": total_still_processing,
        "details": all_results,
    }

    logger.info(
        f"[poll_all_pending_tts_evaluations] Polling summary | "
        f"processed={total_processed} | failed={total_failed} | "
        f"still_processing={total_still_processing}"
    )

    return summary


def _get_batch_jobs_for_run(
    session: Session,
    run: EvaluationRun,
) -> list[BatchJob]:
    """Find all batch jobs associated with a TTS evaluation run.

    Args:
        session: Database session
        run: The evaluation run

    Returns:
        list[BatchJob]: All batch jobs for this run
    """
    stmt = select(BatchJob).where(
        BatchJob.job_type == "tts_evaluation",
        BatchJob.config["evaluation_run_id"].astext.cast(Integer) == run.id,
    )
    return list(session.exec(stmt).all())


def _dispatch_tts_result_processing(
    run: EvaluationRun,
    batch_job: BatchJob,
    org_id: int,
    provider_name: str,
) -> str:
    """Dispatch TTS result processing to Celery low priority queue.

    Args:
        run: The evaluation run
        batch_job: The batch job record
        org_id: Organization ID
        provider_name: TTS provider/model name

    Returns:
        str: Celery task ID
    """
    celery_task_id = start_low_priority_job(
        function_path=_TTS_RESULT_PROCESSING_PATH,
        project_id=run.project_id,
        job_id=str(batch_job.id),
        organization_id=org_id,
        evaluation_run_id=run.id,
        tts_provider=provider_name,
        provider_batch_id=batch_job.provider_batch_id,
    )

    logger.info(
        f"[_dispatch_tts_result_processing] Dispatched to Celery | "
        f"run_id={run.id}, batch_job_id={batch_job.id}, "
        f"provider={provider_name}, celery_task_id={celery_task_id}"
    )

    return celery_task_id


async def poll_tts_run(
    session: Session,
    run: EvaluationRun,
    batch_provider: GeminiBatchProvider,
    org_id: int,
) -> dict[str, Any]:
    """Poll a single TTS evaluation run's batch status.

    Finds all batch jobs for this run (one per provider) and polls each.
    When a batch reaches SUCCEEDED, dispatches result processing to Celery.

    Args:
        session: Database session
        run: The evaluation run to poll
        batch_provider: Initialized GeminiBatchProvider
        org_id: Organization ID

    Returns:
        dict: Status result with run details and action taken
    """
    log_prefix = f"[org={org_id}][project={run.project_id}][eval={run.id}]"
    logger.info(f"[poll_tts_run] {log_prefix} Polling run")

    previous_status = run.status

    batch_jobs = _get_batch_jobs_for_run(session=session, run=run)

    if not batch_jobs:
        logger.warning(f"[poll_tts_run] {log_prefix} No batch jobs found")
        update_tts_run(
            session=session,
            run_id=run.id,
            status="failed",
            error_message="No batch jobs found",
        )
        return {
            "run_id": run.id,
            "run_name": run.run_name,
            "type": "tts",
            "previous_status": previous_status,
            "current_status": "failed",
            "action": "failed",
            "error": "No batch jobs found",
        }

    all_terminal = True
    any_succeeded = False
    any_failed = False
    dispatched = False
    errors: list[str] = []

    for batch_job in batch_jobs:
        provider_name = batch_job.config.get("tts_provider", "unknown")

        # Handle batch jobs already in terminal state
        if batch_job.provider_status in TERMINAL_STATES:
            if batch_job.provider_status == BatchJobState.SUCCEEDED.value:
                # Check if there are still unprocessed results for this batch.
                # This handles retries when a previous processing attempt failed.
                pending_results = get_pending_results_for_run(
                    session=session, run_id=run.id, provider=provider_name
                )
                if pending_results:
                    logger.info(
                        f"[poll_tts_run] {log_prefix} Dispatching reprocessing for "
                        f"{len(pending_results)} pending results | "
                        f"batch_job_id={batch_job.id}"
                    )
                    _dispatch_tts_result_processing(
                        run=run,
                        batch_job=batch_job,
                        org_id=org_id,
                        provider_name=provider_name,
                    )
                    dispatched = True
                any_succeeded = True
            else:
                any_failed = True
                errors.append(
                    f"{provider_name}: {batch_job.error_message or batch_job.provider_status}"
                )
            continue

        # Poll batch job status
        poll_batch_status(
            session=session,
            provider=batch_provider,
            batch_job=batch_job,
        )

        session.refresh(batch_job)
        provider_status = batch_job.provider_status

        logger.info(
            f"[poll_tts_run] {log_prefix} Batch status | "
            f"batch_job_id={batch_job.id} | provider={provider_name} | "
            f"state={provider_status}"
        )

        if provider_status not in TERMINAL_STATES:
            all_terminal = False
            continue

        # Batch reached terminal state - dispatch processing to Celery
        if provider_status == BatchJobState.SUCCEEDED.value:
            _dispatch_tts_result_processing(
                run=run,
                batch_job=batch_job,
                org_id=org_id,
                provider_name=provider_name,
            )
            any_succeeded = True
            dispatched = True
        else:
            any_failed = True
            errors.append(
                f"{provider_name}: {batch_job.error_message or provider_status}"
            )

    if not all_terminal:
        return {
            "run_id": run.id,
            "run_name": run.run_name,
            "type": "tts",
            "previous_status": previous_status,
            "current_status": run.status,
            "action": "no_change",
        }

    # If we dispatched processing to Celery, keep the run as "processing".
    # The Celery task will finalize the run status when done.
    if dispatched:
        return {
            "run_id": run.id,
            "run_name": run.run_name,
            "type": "tts",
            "previous_status": previous_status,
            "current_status": "processing",
            "action": "dispatched",
        }

    # All batch jobs are done and no dispatching needed - finalize the run
    status_counts = count_results_by_status(session=session, run_id=run.id)
    pending = status_counts.get(JobStatus.PENDING.value, 0)
    failed_count = status_counts.get(JobStatus.FAILED.value, 0)

    final_status = "completed" if pending == 0 else "processing"
    error_message = None
    if any_failed:
        error_message = "; ".join(errors)
    elif failed_count > 0:
        error_message = f"{failed_count} synthesis(es) failed"

    update_tts_run(
        session=session,
        run_id=run.id,
        status=final_status,
        error_message=error_message,
    )

    action = "completed" if not any_failed else "failed"

    return {
        "run_id": run.id,
        "run_name": run.run_name,
        "type": "tts",
        "previous_status": previous_status,
        "current_status": final_status,
        "action": action,
        **({"error": error_message} if error_message else {}),
    }
