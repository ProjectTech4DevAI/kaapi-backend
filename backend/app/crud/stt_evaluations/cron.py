"""Cron processing functions for STT evaluations.

This module provides functions that are called periodically to process
pending STT evaluations - polling batch status and processing completed batches.

Follows the same pattern as text evaluations: single query to fetch all
processing runs, grouped by project_id for credential management.
"""

import logging
from typing import Any

from sqlmodel import Session

from app.celery.utils import start_low_priority_job
from app.core.batch import (
    BATCH_KEY,
    GeminiBatchProvider,
    extract_text_from_response_dict,
)
from app.models.batch_job import BatchJobType
from app.core.util import now
from app.crud.evaluations.cron_utils import (
    get_batch_jobs_for_run,
    make_poll_result,
    poll_all_pending_evaluations_by_type,
    poll_batch_jobs,
)
from app.crud.stt_evaluations.result import count_results_by_status
from app.crud.stt_evaluations.run import update_stt_run
from app.models import EvaluationRun
from app.models.batch_job import BatchJob
from app.models.job import JobStatus
from app.models.stt_evaluation import EvaluationType, STTResult

logger = logging.getLogger(__name__)


async def poll_all_pending_stt_evaluations(
    session: Session,
) -> dict[str, Any]:
    """Poll all pending STT evaluations across all organizations.

    Args:
        session: Database session

    Returns:
        Summary dict with total, processed, failed, still_processing counts
    """
    return await poll_all_pending_evaluations_by_type(
        session,
        eval_type="stt",
        eval_type_enum_value=EvaluationType.STT.value,
        update_run_fn=update_stt_run,
        poll_run_fn=poll_stt_run,
        success_actions=("completed", "processed"),
    )


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
    batch_jobs = get_batch_jobs_for_run(
        session=session, run=run, job_type=BatchJobType.STT_EVALUATION
    )

    if not batch_jobs:
        logger.warning(f"[poll_stt_run] No batch jobs found | run_id: {run.id}")
        update_stt_run(
            session=session,
            run_id=run.id,
            status="failed",
            error_message="No batch jobs found",
        )
        return make_poll_result(
            run=run,
            eval_type="stt",
            previous_status=previous_status,
            current_status="failed",
            action="failed",
            error="No batch jobs found",
        )

    async def _on_batch_succeeded(batch_job: BatchJob, provider_name: str) -> bool:
        await process_completed_stt_batch(session, run, batch_job, batch_provider)
        return False

    result = await poll_batch_jobs(
        session=session,
        batch_jobs=batch_jobs,
        batch_provider=batch_provider,
        provider_config_key="stt_provider",
        log_prefix=f"[poll_stt_run] run_id={run.id}",
        on_succeeded=_on_batch_succeeded,
    )

    if not result.all_terminal:
        return make_poll_result(
            run=run,
            eval_type="stt",
            previous_status=previous_status,
            current_status=run.status,
            action="no_change",
        )

    # All batch jobs are done - clean up Gemini audio files once
    gemini_file_ids = []
    for bj in batch_jobs:
        gemini_file_ids = bj.config.get("gemini_audio_file_ids", [])
        if gemini_file_ids:
            break

    if gemini_file_ids:
        try:
            deleted, failed = batch_provider.delete_files(gemini_file_ids)
            logger.info(
                f"[poll_stt_run] Gemini file cleanup | "
                f"run_id={run.id}, deleted={deleted}, failed={failed}"
            )
        except Exception as e:
            # Non-critical; Gemini files auto-expire after 48h
            logger.warning(
                f"[poll_stt_run] Gemini file cleanup failed | "
                f"run_id={run.id}, error={str(e)}"
            )

    # Finalize the run
    status_counts = count_results_by_status(session=session, run_id=run.id)
    failed_count = status_counts.get(JobStatus.FAILED.value, 0)

    final_status = "completed"
    error_message = None
    if result.any_failed:
        error_message = "; ".join(result.errors)
    elif failed_count > 0:
        error_message = f"{failed_count} transcription(s) failed"

    update_stt_run(
        session=session,
        run_id=run.id,
        status=final_status,
        error_message=error_message,
    )

    # Trigger automated metric computation (WER, CER, lenient WER, WIP)
    if result.any_succeeded:
        try:
            celery_task_id = start_low_priority_job(
                function_path="app.services.stt_evaluations.metric_job.execute_metric_computation",
                project_id=run.project_id,
                job_id=str(run.id),
                organization_id=run.organization_id,
                run_id=run.id,
            )
            logger.info(
                f"[poll_stt_run] Metric computation task dispatched | "
                f"run_id: {run.id}, celery_task_id: {celery_task_id}"
            )
        except Exception as e:
            logger.error(
                f"[poll_stt_run] Failed to dispatch metric computation task | "
                f"run_id: {run.id}, error: {e}",
                exc_info=True,
            )

    action = "completed" if not result.any_failed else "failed"

    return make_poll_result(
        run=run,
        eval_type="stt",
        previous_status=previous_status,
        current_status=final_status,
        action=action,
        error=error_message,
    )


async def process_completed_stt_batch(
    session: Session,
    run: EvaluationRun,
    batch_job: BatchJob,
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
        stt_result_rows: list[dict[str, Any]] = []

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
                row["transcription"] = extract_text_from_response_dict(
                    response["response"]
                )
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
