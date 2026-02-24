"""Cron processing functions for TTS evaluations.

This module provides functions that are called periodically to process
pending TTS evaluations - polling batch status and dispatching result
processing to Celery workers.

Follows the same pattern as STT evaluations: single query to fetch all
processing runs, grouped by project_id for credential management.
"""

import logging
from typing import Any

from sqlmodel import Session

from app.celery.utils import start_low_priority_job
from app.core.batch import (
    BatchJobState,
    GeminiBatchProvider,
    poll_batch_status,
)
from app.crud.evaluations.cron_utils import (
    TERMINAL_STATES,
    get_batch_jobs_for_run,
    poll_all_pending_evaluations_by_type,
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

logger = logging.getLogger(__name__)

# Function path for Celery task dispatch
_TTS_RESULT_PROCESSING_PATH = (
    "app.services.tts_evaluations.batch_result_processing.execute_tts_result_processing"
)


async def poll_all_pending_tts_evaluations(
    session: Session,
) -> dict[str, Any]:
    """Poll all pending TTS evaluations across all organizations.

    Args:
        session: Database session

    Returns:
        Summary dict with total, processed, failed, still_processing counts
    """
    return await poll_all_pending_evaluations_by_type(
        session,
        eval_type="tts",
        eval_type_enum_value=EvaluationType.TTS.value,
        update_run_fn=update_tts_run,
        poll_run_fn=poll_tts_run,
        success_actions=("completed", "processed", "dispatched"),
    )


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

    batch_jobs = get_batch_jobs_for_run(
        session=session, run=run, job_type="tts_evaluation"
    )

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
