"""Cron processing functions for STT evaluations.

This module provides functions that are called periodically to process
pending STT evaluations - polling batch status and processing completed batches.
"""

import logging
from typing import Any

from sqlalchemy import Integer
from sqlmodel import Session, select

from app.core.batch import BatchJobState, GeminiBatchProvider, poll_batch_status
from app.crud.stt_evaluations.result import count_results_by_status, update_stt_result
from app.crud.stt_evaluations.run import get_pending_stt_runs, update_stt_run
from app.models import EvaluationRun
from app.models.batch_job import BatchJob
from app.models.stt_evaluation import STTResult, STTResultStatus
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
    org_id: int,
) -> dict[str, Any]:
    """Poll all pending STT evaluations for an organization.

    This function:
    1. Gets all STT runs in "processing" status
    2. For each run, polls the Gemini batch status
    3. If completed, processes the results
    4. Returns a summary of what was processed

    Args:
        session: Database session
        org_id: Organization ID

    Returns:
        dict: Summary with processed, failed, still_processing counts
    """
    logger.info(
        f"[poll_all_pending_stt_evaluations] Starting STT evaluation polling | "
        f"org_id: {org_id}"
    )

    # Get all pending STT runs for this organization
    pending_runs = get_pending_stt_runs(session=session, org_id=org_id)

    if not pending_runs:
        logger.info(
            f"[poll_all_pending_stt_evaluations] No pending STT runs | org_id: {org_id}"
        )
        return {"processed": 0, "failed": 0, "still_processing": 0}

    logger.info(
        f"[poll_all_pending_stt_evaluations] Found {len(pending_runs)} pending STT runs | "
        f"org_id: {org_id}"
    )

    processed = 0
    failed = 0
    still_processing = 0

    for run in pending_runs:
        try:
            result = await poll_stt_run(session=session, run=run, org_id=org_id)

            if result["status"] == "completed":
                processed += 1
            elif result["status"] == "failed":
                failed += 1
            else:  # still_processing
                still_processing += 1

        except Exception as e:
            logger.error(
                f"[poll_all_pending_stt_evaluations] Error polling run | "
                f"run_id: {run.id}, error: {str(e)}",
                exc_info=True,
            )
            failed += 1

    logger.info(
        f"[poll_all_pending_stt_evaluations] Polling complete | "
        f"org_id: {org_id}, processed: {processed}, failed: {failed}, "
        f"still_processing: {still_processing}"
    )

    return {
        "processed": processed,
        "failed": failed,
        "still_processing": still_processing,
    }


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
    org_id: int,
) -> dict[str, Any]:
    """Poll a single STT evaluation run's batch status.

    Finds all batch jobs for this run (one per provider) and polls each.
    Only marks the run as complete when all batch jobs are in terminal states.

    Args:
        session: Database session
        run: The evaluation run to poll
        org_id: Organization ID

    Returns:
        dict: Status result with "status" key (completed/failed/still_processing)
    """
    logger.info(f"[poll_stt_run] Polling run | run_id: {run.id}")

    # Find all batch jobs for this run
    batch_jobs = _get_batch_jobs_for_run(session=session, run=run)

    if not batch_jobs:
        logger.warning(f"[poll_stt_run] No batch jobs found | run_id: {run.id}")
        return {"status": "failed", "error": "No batch jobs found"}

    try:
        # Initialize Gemini client
        gemini_client = GeminiClient.from_credentials(
            session=session,
            org_id=org_id,
            project_id=run.project_id,
        )
        batch_provider = GeminiBatchProvider(client=gemini_client.client)

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
                f"[poll_stt_run] Batch status | "
                f"run_id: {run.id}, batch_job_id: {batch_job.id}, "
                f"provider: {provider_name}, state: {provider_status}"
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
                    org_id=org_id,
                )
                any_succeeded = True
            else:
                any_failed = True
                errors.append(
                    f"{provider_name}: {batch_job.error_message or provider_status}"
                )

        if not all_terminal:
            return {"status": "still_processing"}

        # All batch jobs are done - finalize the run
        status_counts = count_results_by_status(session=session, run_id=run.id)
        pending = status_counts.get(STTResultStatus.PENDING.value, 0)
        failed = status_counts.get(STTResultStatus.FAILED.value, 0)

        final_status = "completed" if pending == 0 else "processing"
        error_message = None
        if any_failed:
            error_message = "; ".join(errors)
        elif failed > 0:
            error_message = f"{failed} transcription(s) failed"

        update_stt_run(
            session=session,
            run_id=run.id,
            status=final_status,
            error_message=error_message,
        )

        return {"status": "completed" if not any_failed else "failed", "errors": errors}

    except Exception as e:
        logger.error(
            f"[poll_stt_run] Error polling run | run_id: {run.id}, error: {str(e)}",
            exc_info=True,
        )
        return {"status": "failed", "error": str(e)}


async def process_completed_stt_batch(
    session: Session,
    run: EvaluationRun,
    batch_job: Any,
    org_id: int,
) -> None:
    """Process completed Gemini batch - download results and update STT result records.

    Args:
        session: Database session
        run: The evaluation run
        batch_job: The BatchJob record
        org_id: Organization ID
    """
    logger.info(
        f"[process_completed_stt_batch] Processing batch results | "
        f"run_id: {run.id}, batch_job_id: {batch_job.id}"
    )

    # Get the STT provider from batch job config
    stt_provider = batch_job.config.get("stt_provider", "gemini-2.5-pro")

    # Initialize Gemini client
    gemini_client = GeminiClient.from_credentials(
        session=session,
        org_id=org_id,
        project_id=run.project_id,
    )
    batch_provider = GeminiBatchProvider(client=gemini_client.client)

    processed_count = 0
    failed_count = 0

    try:
        # Download results using GeminiBatchProvider
        # Keys are embedded in the JSONL response file, no separate mapping needed
        results = batch_provider.download_batch_results(batch_job.provider_batch_id)

        logger.info(
            f"[process_completed_stt_batch] Got batch results | "
            f"batch_job_id: {batch_job.id}, result_count: {len(results)}"
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
                    f"batch_job_id: {batch_job.id}, custom_id: {custom_id}"
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
                        status=STTResultStatus.COMPLETED.value,
                    )
                    processed_count += 1
                else:
                    update_stt_result(
                        session=session,
                        result_id=result_record.id,
                        status=STTResultStatus.FAILED.value,
                        error_message=batch_result.get("error", "Unknown error"),
                    )
                    failed_count += 1

    except Exception as e:
        logger.error(
            f"[process_completed_stt_batch] Failed to process batch results | "
            f"batch_job_id: {batch_job.id}, error: {str(e)}",
            exc_info=True,
        )
        raise

    logger.info(
        f"[process_completed_stt_batch] Batch results processed | "
        f"run_id: {run.id}, provider: {stt_provider}, "
        f"processed: {processed_count}, failed: {failed_count}"
    )
