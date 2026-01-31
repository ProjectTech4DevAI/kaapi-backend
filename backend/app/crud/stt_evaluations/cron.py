"""Cron processing functions for STT evaluations.

This module provides functions that are called periodically to process
pending STT evaluations - polling batch status and processing completed batches.
"""

import logging
from typing import Any

from sqlmodel import Session, select

from app.core.batch import BatchJobState, GeminiBatchProvider
from app.crud.stt_evaluations.result import count_results_by_status, update_stt_result
from app.crud.stt_evaluations.run import get_pending_stt_runs, update_stt_run
from app.models import EvaluationRun
from app.models.stt_evaluation import STTResult, STTResultStatus
from app.services.stt_evaluations.gemini import GeminiClient, GeminiFilesManager

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


async def poll_stt_run(
    session: Session,
    run: EvaluationRun,
    org_id: int,
) -> dict[str, Any]:
    """Poll a single STT evaluation run's batch status.

    Args:
        session: Database session
        run: The evaluation run to poll
        org_id: Organization ID

    Returns:
        dict: Status result with "status" key (completed/failed/still_processing)
    """
    logger.info(f"[poll_stt_run] Polling run | run_id: {run.id}")

    # Check if run has batch jobs stored
    if not run.score or "batch_jobs" not in run.score:
        logger.warning(
            f"[poll_stt_run] Run has no batch_jobs in score | run_id: {run.id}"
        )
        return {"status": "failed", "error": "No batch jobs found"}

    batch_jobs: dict[str, str] = run.score.get("batch_jobs", {})

    if not batch_jobs:
        logger.warning(f"[poll_stt_run] Empty batch_jobs | run_id: {run.id}")
        return {"status": "failed", "error": "Empty batch jobs"}

    try:
        # Initialize Gemini client
        gemini_client = GeminiClient.from_credentials(
            session=session,
            org_id=org_id,
            project_id=run.project_id,
        )
        batch_provider = GeminiBatchProvider(client=gemini_client.client)

        all_complete = True
        any_success = False

        for provider, batch_id in batch_jobs.items():
            status = batch_provider.get_batch_status(batch_id)
            provider_status = status["provider_status"]

            logger.info(
                f"[poll_stt_run] Batch status | "
                f"run_id: {run.id}, provider: {provider}, "
                f"batch_id: {batch_id}, state: {provider_status}"
            )

            is_terminal = provider_status in TERMINAL_STATES

            if not is_terminal:
                all_complete = False
            elif provider_status == BatchJobState.SUCCEEDED.value:
                any_success = True

        if not all_complete:
            return {"status": "still_processing"}

        # All batches complete - process results
        if any_success:
            await process_completed_stt_batch(
                session=session,
                run=run,
                batch_jobs=batch_jobs,
                org_id=org_id,
            )
            return {"status": "completed"}
        else:
            update_stt_run(
                session=session,
                run_id=run.id,
                status="failed",
                error_message="All batch jobs failed",
            )
            return {"status": "failed", "error": "All batch jobs failed"}

    except Exception as e:
        logger.error(
            f"[poll_stt_run] Error polling run | run_id: {run.id}, error: {str(e)}",
            exc_info=True,
        )
        return {"status": "failed", "error": str(e)}


async def process_completed_stt_batch(
    session: Session,
    run: EvaluationRun,
    batch_jobs: dict[str, str],
    org_id: int,
) -> None:
    """Process completed Gemini batch - download results and update STT result records.

    Args:
        session: Database session
        run: The evaluation run
        batch_jobs: Dict of provider -> batch_id
        org_id: Organization ID
    """
    logger.info(
        f"[process_completed_stt_batch] Processing batch results | run_id: {run.id}"
    )

    sample_file_mapping = run.score.get("sample_file_mapping", [])
    sample_ids = [item["sample_id"] for item in sample_file_mapping]

    # Initialize Gemini client and providers
    gemini_client = GeminiClient.from_credentials(
        session=session,
        org_id=org_id,
        project_id=run.project_id,
    )
    batch_provider = GeminiBatchProvider(client=gemini_client.client)
    files_manager = GeminiFilesManager(gemini_client.client)

    processed_count = 0
    failed_count = 0

    for provider, batch_id in batch_jobs.items():
        try:
            # Check if batch succeeded before downloading
            status = batch_provider.get_batch_status(batch_id)
            if status["provider_status"] != BatchJobState.SUCCEEDED.value:
                logger.warning(
                    f"[process_completed_stt_batch] Batch not succeeded | "
                    f"provider: {provider}, status: {status['provider_status']}"
                )
                # Mark results for this provider as failed
                for item in sample_file_mapping:
                    sample_id = item["sample_id"]
                    stmt = select(STTResult).where(
                        STTResult.evaluation_run_id == run.id,
                        STTResult.stt_sample_id == sample_id,
                        STTResult.provider == provider,
                    )
                    result_record = session.exec(stmt).one_or_none()
                    if result_record:
                        update_stt_result(
                            session=session,
                            result_id=result_record.id,
                            status=STTResultStatus.FAILED.value,
                            error_message=f"Batch {status['provider_status']}",
                        )
                        failed_count += 1
                continue

            # Download results using GeminiBatchProvider
            results = batch_provider.download_batch_results(batch_id)

            logger.info(
                f"[process_completed_stt_batch] Got batch results | "
                f"provider: {provider}, result_count: {len(results)}"
            )

            # Match results to samples by index
            for batch_result in results:
                custom_id = batch_result["custom_id"]
                # custom_id is the index as string
                try:
                    index = int(custom_id)
                except (ValueError, TypeError):
                    index = results.index(batch_result)

                if index >= len(sample_ids):
                    continue

                sample_id = sample_ids[index]

                # Find result record for this sample and provider
                stmt = select(STTResult).where(
                    STTResult.evaluation_run_id == run.id,
                    STTResult.stt_sample_id == sample_id,
                    STTResult.provider == provider,
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
                f"[process_completed_stt_batch] Failed to process provider results | "
                f"provider: {provider}, error: {str(e)}"
            )
            failed_count += len(sample_file_mapping)

    # Clean up Google Files
    for item in sample_file_mapping:
        file_uri = item.get("file_uri")
        if file_uri:
            try:
                files_manager.delete_file(file_uri)
            except Exception as e:
                logger.warning(
                    f"[process_completed_stt_batch] Failed to delete Google file | "
                    f"file_uri: {file_uri}, error: {str(e)}"
                )

    # Update run status
    status_counts = count_results_by_status(session=session, run_id=run.id)

    completed = status_counts.get(STTResultStatus.COMPLETED.value, 0)
    failed = status_counts.get(STTResultStatus.FAILED.value, 0)
    pending = status_counts.get(STTResultStatus.PENDING.value, 0)

    final_status = "completed" if pending == 0 else "processing"
    error_message = None
    if failed > 0:
        error_message = f"{failed} transcription(s) failed"

    update_stt_run(
        session=session,
        run_id=run.id,
        status=final_status,
        processed_samples=completed + failed,
        error_message=error_message,
    )

    logger.info(
        f"[process_completed_stt_batch] Batch results processed | "
        f"run_id: {run.id}, completed: {completed}, "
        f"failed: {failed}, status: {final_status}"
    )
