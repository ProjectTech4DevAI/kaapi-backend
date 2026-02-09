"""Cron processing functions for STT evaluations.

This module provides functions that are called periodically to process
pending STT evaluations - polling batch status and processing completed batches.

Follows the same pattern as text evaluations: single query to fetch all
processing runs, grouped by project_id for credential management.
"""

import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import Integer
from sqlmodel import Session, select

from app.core.batch import BatchJobState, GeminiBatchProvider, poll_batch_status
from app.crud.stt_evaluations.result import count_results_by_status, update_stt_result
from app.crud.stt_evaluations.run import update_stt_run
from app.models import EvaluationRun
from app.models.batch_job import BatchJob
from app.models.job import JobStatus
from app.models.stt_evaluation import EvaluationType, STTResult
from app.services.stt_evaluations.gemini import GeminiClient

logger = logging.getLogger(__name__)

# Terminal states that indicate batch processing is complete
TERMINAL_STATES = {
    BatchJobState.SUCCEEDED.value,
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
}


async def poll_all_pending_stt_evaluations(
    session: Session,
) -> dict[str, Any]:
    """Poll all pending STT evaluations across all organizations.

    Fetches all STT evaluation runs with status='processing' in a single query,
    groups them by project_id, and processes each project with its own
    Gemini client.

    Args:
        session: Database session

    Returns:
        Summary dict:
        {
            "total": 5,
            "processed": 2,
            "failed": 1,
            "still_processing": 2,
            "details": [...]
        }
    """
    logger.info("[poll_all_pending_stt_evaluations] Starting STT evaluation polling")

    # Single query to fetch all processing STT evaluation runs
    statement = select(EvaluationRun).where(
        EvaluationRun.type == EvaluationType.STT.value,
        EvaluationRun.status == "processing",
        EvaluationRun.batch_job_id.is_not(None),
    )
    pending_runs = session.exec(statement).all()

    if not pending_runs:
        logger.info("[poll_all_pending_stt_evaluations] No pending STT runs found")
        return {
            "total": 0,
            "processed": 0,
            "failed": 0,
            "still_processing": 0,
            "details": [],
        }

    logger.info(
        f"[poll_all_pending_stt_evaluations] Found {len(pending_runs)} pending STT runs"
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
        # All runs in a project share the same org_id
        org_id = project_runs[0].organization_id

        try:
            # Initialize Gemini client for this project
            try:
                gemini_client = GeminiClient.from_credentials(
                    session=session,
                    org_id=org_id,
                    project_id=project_id,
                )
            except Exception as client_err:
                logger.error(
                    f"[poll_all_pending_stt_evaluations] Failed to get Gemini client | "
                    f"org_id={org_id} | project_id={project_id} | error={client_err}"
                )
                # Mark all runs in this project as failed
                for run in project_runs:
                    update_stt_run(
                        session=session,
                        run_id=run.id,
                        status="failed",
                        error_message=f"Gemini client initialization failed: {str(client_err)}",
                    )
                    all_results.append(
                        {
                            "run_id": run.id,
                            "run_name": run.run_name,
                            "type": "stt",
                            "action": "failed",
                            "error": str(client_err),
                        }
                    )
                    total_failed += 1
                continue

            batch_provider = GeminiBatchProvider(client=gemini_client.client)

            # Process each run in this project
            for run in project_runs:
                try:
                    result = await poll_stt_run(
                        session=session,
                        run=run,
                        batch_provider=batch_provider,
                        org_id=org_id,
                    )
                    all_results.append(result)

                    if result["action"] in ("completed", "processed"):
                        total_processed += 1
                    elif result["action"] == "failed":
                        total_failed += 1
                    else:
                        total_still_processing += 1

                except Exception as e:
                    logger.error(
                        f"[poll_all_pending_stt_evaluations] Failed to poll STT run | "
                        f"run_id={run.id} | {e}",
                        exc_info=True,
                    )
                    update_stt_run(
                        session=session,
                        run_id=run.id,
                        status="failed",
                        error_message=f"Polling failed: {str(e)}",
                    )
                    all_results.append(
                        {
                            "run_id": run.id,
                            "run_name": run.run_name,
                            "type": "stt",
                            "action": "failed",
                            "error": str(e),
                        }
                    )
                    total_failed += 1

        except Exception as e:
            logger.error(
                f"[poll_all_pending_stt_evaluations] Failed to process project | "
                f"project_id={project_id} | {e}",
                exc_info=True,
            )
            for run in project_runs:
                update_stt_run(
                    session=session,
                    run_id=run.id,
                    status="failed",
                    error_message=f"Project processing failed: {str(e)}",
                )
                all_results.append(
                    {
                        "run_id": run.id,
                        "run_name": run.run_name,
                        "type": "stt",
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
        f"[poll_all_pending_stt_evaluations] Polling summary | "
        f"processed={total_processed} | failed={total_failed} | "
        f"still_processing={total_still_processing}"
    )

    return summary


def _get_batch_jobs_for_run(
    session: Session,
    run: EvaluationRun,
) -> list[BatchJob]:
    """Find all batch jobs associated with an STT evaluation run.

    Queries batch_job table where config contains the evaluation_run_id.

    Args:
        session: Database session
        run: The evaluation run

    Returns:
        list[BatchJob]: All batch jobs for this run
    """
    stmt = select(BatchJob).where(
        BatchJob.job_type == "stt_evaluation",
        BatchJob.config["evaluation_run_id"].astext.cast(Integer) == run.id,
    )
    return list(session.exec(stmt).all())


async def poll_stt_run(
    session: Session,
    run: EvaluationRun,
    batch_provider: GeminiBatchProvider,
    org_id: int,
) -> dict[str, Any]:
    """Poll a single STT evaluation run's batch status.

    Finds all batch jobs for this run (one per provider) and polls each.
    Only marks the run as complete when all batch jobs are in terminal states.

    Args:
        session: Database session
        run: The evaluation run to poll
        batch_provider: Initialized GeminiBatchProvider
        org_id: Organization ID

    Returns:
        dict: Status result with run details and action taken
    """
    log_prefix = f"[org={org_id}][project={run.project_id}][eval={run.id}]"
    logger.info(f"[poll_stt_run] {log_prefix} Polling run")

    previous_status = run.status

    # Find all batch jobs for this run
    batch_jobs = _get_batch_jobs_for_run(session=session, run=run)

    if not batch_jobs:
        logger.warning(f"[poll_stt_run] {log_prefix} No batch jobs found")
        update_stt_run(
            session=session,
            run_id=run.id,
            status="failed",
            error_message="No batch jobs found",
        )
        return {
            "run_id": run.id,
            "run_name": run.run_name,
            "type": "stt",
            "previous_status": previous_status,
            "current_status": "failed",
            "action": "failed",
            "error": "No batch jobs found",
        }

    all_terminal = True
    any_succeeded = False
    any_failed = False
    errors: list[str] = []

    for batch_job in batch_jobs:
        provider_name = batch_job.config.get("stt_provider", "unknown")

        # Skip batch jobs already in terminal state that have been processed
        if batch_job.provider_status in TERMINAL_STATES:
            if batch_job.provider_status == BatchJobState.SUCCEEDED.value:
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
            f"[poll_stt_run] {log_prefix} Batch status | "
            f"batch_job_id={batch_job.id} | provider={provider_name} | "
            f"state={provider_status}"
        )

        if provider_status not in TERMINAL_STATES:
            all_terminal = False
            continue

        # Batch reached terminal state - process it
        if provider_status == BatchJobState.SUCCEEDED.value:
            await process_completed_stt_batch(
                session=session,
                run=run,
                batch_job=batch_job,
                batch_provider=batch_provider,
            )
            any_succeeded = True
        else:
            any_failed = True
            errors.append(
                f"{provider_name}: {batch_job.error_message or provider_status}"
            )

    if not all_terminal:
        return {
            "run_id": run.id,
            "run_name": run.run_name,
            "type": "stt",
            "previous_status": previous_status,
            "current_status": run.status,
            "action": "no_change",
        }

    # All batch jobs are done - finalize the run
    status_counts = count_results_by_status(session=session, run_id=run.id)
    pending = status_counts.get(JobStatus.PENDING.value, 0)
    failed_count = status_counts.get(JobStatus.FAILED.value, 0)

    final_status = "completed" if pending == 0 else "processing"
    error_message = None
    if any_failed:
        error_message = "; ".join(errors)
    elif failed_count > 0:
        error_message = f"{failed_count} transcription(s) failed"

    update_stt_run(
        session=session,
        run_id=run.id,
        status=final_status,
        error_message=error_message,
    )

    action = "completed" if not any_failed else "failed"

    return {
        "run_id": run.id,
        "run_name": run.run_name,
        "type": "stt",
        "previous_status": previous_status,
        "current_status": final_status,
        "action": action,
        **({"error": error_message} if error_message else {}),
    }


async def process_completed_stt_batch(
    session: Session,
    run: EvaluationRun,
    batch_job: Any,
    batch_provider: GeminiBatchProvider,
) -> None:
    """Process completed Gemini batch - download results and update STT result records.

    Args:
        session: Database session
        run: The evaluation run
        batch_job: The BatchJob record
        batch_provider: Initialized GeminiBatchProvider
    """
    logger.info(
        f"[process_completed_stt_batch] Processing batch results | "
        f"run_id={run.id}, batch_job_id={batch_job.id}"
    )

    # Get the STT provider from batch job config
    stt_provider = batch_job.config.get("stt_provider", "gemini-2.5-pro")

    processed_count = 0
    failed_count = 0

    try:
        # Download results using GeminiBatchProvider
        # Keys are embedded in the JSONL response file, no separate mapping needed
        results = batch_provider.download_batch_results(batch_job.provider_batch_id)

        logger.info(
            f"[process_completed_stt_batch] Got batch results | "
            f"batch_job_id={batch_job.id}, result_count={len(results)}"
        )

        # Match results to samples using key (sample_id) from batch request
        for batch_result in results:
            custom_id = batch_result["custom_id"]
            # custom_id is the sample_id as string (set via key in batch request)
            try:
                sample_id = int(custom_id)
            except (ValueError, TypeError):
                logger.warning(
                    f"[process_completed_stt_batch] Invalid custom_id | "
                    f"batch_job_id={batch_job.id}, custom_id={custom_id}"
                )
                failed_count += 1
                continue

            # Find result record for this sample and provider
            stmt = select(STTResult).where(
                STTResult.evaluation_run_id == run.id,
                STTResult.stt_sample_id == sample_id,
                STTResult.provider == stt_provider,
            )
            result_record = session.exec(stmt).one_or_none()

            if result_record:
                if batch_result.get("response"):
                    text = batch_result["response"].get("text", "")
                    update_stt_result(
                        session=session,
                        result_id=result_record.id,
                        transcription=text,
                        status=JobStatus.SUCCESS.value,
                    )
                    processed_count += 1
                else:
                    update_stt_result(
                        session=session,
                        result_id=result_record.id,
                        status=JobStatus.FAILED.value,
                        error_message=batch_result.get("error", "Unknown error"),
                    )
                    failed_count += 1

        session.commit()

    except Exception as e:
        logger.error(
            f"[process_completed_stt_batch] Failed to process batch results | "
            f"batch_job_id={batch_job.id}, error={str(e)}",
            exc_info=True,
        )
        raise

    logger.info(
        f"[process_completed_stt_batch] Batch results processed | "
        f"run_id={run.id}, provider={stt_provider}, "
        f"processed={processed_count}, failed={failed_count}"
    )
