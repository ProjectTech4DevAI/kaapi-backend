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
from app.core.batch.base import BATCH_KEY
from app.core.util import now
from app.crud.stt_evaluations.result import count_results_by_status
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
            gemini_client = GeminiClient.from_credentials(
                session=session,
                org_id=org_id,
                project_id=project_id,
            )
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
                        f"run_id: {run.id}, error: {e}",
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
                f"project_id: {project_id}, error: {e}",
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
            total_failed += len(project_runs)

    summary = {
        "total": len(pending_runs),
        "processed": total_processed,
        "failed": total_failed,
        "still_processing": total_still_processing,
        "details": all_results,
    }

    logger.info(
        f"[poll_all_pending_stt_evaluations] Polling summary | "
        f"processed: {total_processed}, failed: {total_failed}, "
        f"still_processing: {total_still_processing}"
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
    logger.info(
        f"[poll_stt_run] Polling run | "
        f"run_id: {run.id}, org_id: {org_id}, project_id: {run.project_id}"
    )

    previous_status = run.status

    # Find all batch jobs for this run
    batch_jobs = _get_batch_jobs_for_run(session=session, run=run)

    if not batch_jobs:
        logger.warning(f"[poll_stt_run] No batch jobs found | run_id: {run.id}")
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
    failed_count = status_counts.get(JobStatus.FAILED.value, 0)

    final_status = "completed"
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
    """Process completed Gemini batch - download results and create STT result records.

    Result records are created here on batch completion rather than upfront,
    using the stt_sample_id embedded as the key in each batch request.

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

    stt_provider = batch_job.config.get("stt_provider", "gemini-2.5-pro")

    success_count = 0
    failure_count = 0

    try:
        batch_responses = batch_provider.download_batch_results(
            batch_job.provider_batch_id
        )

        logger.info(
            f"[process_completed_stt_batch] Downloaded batch responses | "
            f"batch_job_id={batch_job.id}, response_count={len(batch_responses)}"
        )

        timestamp = now()
        stt_result_rows: list[dict] = []

        for response in batch_responses:
            raw_sample_id = response[BATCH_KEY]
            try:
                stt_sample_id = int(raw_sample_id)
            except (ValueError, TypeError):
                logger.warning(
                    f"[process_completed_stt_batch] Invalid {BATCH_KEY} | "
                    f"batch_job_id={batch_job.id}, {BATCH_KEY}={raw_sample_id}"
                )
                failure_count += 1
                continue

            row = {
                "stt_sample_id": stt_sample_id,
                "evaluation_run_id": run.id,
                "organization_id": run.organization_id,
                "project_id": run.project_id,
                "provider": stt_provider,
                "inserted_at": timestamp,
                "updated_at": timestamp,
            }

            if response.get("response"):
                row["transcription"] = response["response"].get("text", "")
                row["status"] = JobStatus.SUCCESS.value
                success_count += 1
            else:
                row["status"] = JobStatus.FAILED.value
                row["error_message"] = response.get("error", "Unknown error")
                failure_count += 1

            stt_result_rows.append(row)

        # Bulk insert in batches of 200
        insert_batch_size = 200
        for i in range(0, len(stt_result_rows), insert_batch_size):
            chunk = stt_result_rows[i : i + insert_batch_size]
            session.bulk_insert_mappings(STTResult, chunk)
        if stt_result_rows:
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
        f"success={success_count}, failed={failure_count}"
    )
