"""Cron processing functions for TTS evaluations.

This module provides functions that are called periodically to process
pending TTS evaluations - polling batch status and processing completed batches.

Follows the same pattern as STT evaluations: single query to fetch all
processing runs, grouped by project_id for credential management.
"""

import base64
import logging
import uuid
from collections import defaultdict
from typing import Any

from sqlalchemy import Integer
from sqlmodel import Session, select

from app.core.batch import (
    BATCH_KEY,
    BatchJobState,
    GeminiBatchProvider,
    poll_batch_status,
)
from app.core.cloud.storage import get_cloud_storage
from app.core.storage_utils import upload_to_object_store
from app.crud.tts_evaluations.result import count_results_by_status, update_tts_result
from app.crud.tts_evaluations.run import update_tts_run
from app.models import EvaluationRun
from app.models.batch_job import BatchJob
from app.models.job import JobStatus
from app.models.stt_evaluation import EvaluationType
from app.models.tts_evaluation import TTSResult
from app.services.stt_evaluations.gemini import GeminiClient
from app.services.tts_evaluations.audio import calculate_duration, pcm_to_wav

logger = logging.getLogger(__name__)

# Terminal states that indicate batch processing is complete
TERMINAL_STATES = {
    BatchJobState.SUCCEEDED.value,
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
}


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

                    if result["action"] in ("completed", "processed"):
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


async def poll_tts_run(
    session: Session,
    run: EvaluationRun,
    batch_provider: GeminiBatchProvider,
    org_id: int,
) -> dict[str, Any]:
    """Poll a single TTS evaluation run's batch status.

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
    errors: list[str] = []

    for batch_job in batch_jobs:
        provider_name = batch_job.config.get("tts_provider", "unknown")

        # Skip batch jobs already in terminal state
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
            f"[poll_tts_run] {log_prefix} Batch status | "
            f"batch_job_id={batch_job.id} | provider={provider_name} | "
            f"state={provider_status}"
        )

        if provider_status not in TERMINAL_STATES:
            all_terminal = False
            continue

        # Batch reached terminal state - process it
        if provider_status == BatchJobState.SUCCEEDED.value:
            await process_completed_tts_batch(
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
            "type": "tts",
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


async def process_completed_tts_batch(
    session: Session,
    run: EvaluationRun,
    batch_job: Any,
    batch_provider: GeminiBatchProvider,
) -> None:
    """Process completed Gemini batch - download results, convert audio, upload to S3.

    For each result:
    1. Download JSONL results from Gemini
    2. Extract base64-encoded PCM audio from response
    3. Decode base64 -> raw PCM bytes
    4. Wrap in WAV container (24kHz, 16-bit, mono)
    5. Upload WAV to S3
    6. Update TTSResult with object_store_url and metadata

    Args:
        session: Database session
        run: The evaluation run
        batch_job: The BatchJob record
        batch_provider: Initialized GeminiBatchProvider
    """
    logger.info(
        f"[process_completed_tts_batch] Processing batch results | "
        f"run_id={run.id}, batch_job_id={batch_job.id}"
    )

    tts_provider = batch_job.config.get("tts_provider", "gemini-2.5-pro-preview-tts")

    # Get cloud storage for S3 uploads
    storage = get_cloud_storage(session=session, project_id=run.project_id)

    processed_count = 0
    failed_count = 0

    try:
        results = batch_provider.download_batch_results(batch_job.provider_batch_id)

        logger.info(
            f"[process_completed_tts_batch] Got batch results | "
            f"batch_job_id={batch_job.id}, result_count={len(results)}"
        )

        for batch_result in results:
            custom_id = batch_result[BATCH_KEY]
            try:
                result_id = int(custom_id)
            except (ValueError, TypeError):
                logger.warning(
                    f"[process_completed_tts_batch] Invalid {BATCH_KEY} | "
                    f"batch_job_id={batch_job.id}, {BATCH_KEY}={custom_id}"
                )
                failed_count += 1
                continue

            # Find result record
            stmt = select(TTSResult).where(
                TTSResult.id == result_id,
                TTSResult.evaluation_run_id == run.id,
                TTSResult.provider == tts_provider,
            )
            result_record = session.exec(stmt).one_or_none()

            if not result_record:
                logger.warning(
                    f"[process_completed_tts_batch] Result record not found | "
                    f"result_id={result_id}"
                )
                failed_count += 1
                continue

            if batch_result.get("response"):
                try:
                    # Extract base64 audio from Gemini TTS response
                    audio_b64 = _extract_audio_from_response(batch_result["response"])

                    if not audio_b64:
                        update_tts_result(
                            session=session,
                            result_id=result_record.id,
                            status=JobStatus.FAILED.value,
                            error_message="No audio data in response",
                        )
                        failed_count += 1
                        continue

                    # Decode base64 -> raw PCM bytes
                    pcm_data = base64.b64decode(audio_b64)

                    # Wrap in WAV container
                    wav_data = pcm_to_wav(pcm_data)

                    # Calculate duration
                    duration = calculate_duration(len(pcm_data))

                    # Upload WAV to S3
                    audio_filename = f"{uuid.uuid4()}.wav"
                    audio_url = upload_to_object_store(
                        storage=storage,
                        content=wav_data,
                        filename=audio_filename,
                        subdirectory=f"evaluations/tts/audio",
                        content_type="audio/wav",
                    )

                    # Update result
                    update_tts_result(
                        session=session,
                        result_id=result_record.id,
                        object_store_url=audio_url,
                        metadata={
                            "duration_seconds": round(duration, 3),
                            "size_bytes": len(wav_data),
                        },
                        status=JobStatus.SUCCESS.value,
                    )
                    processed_count += 1

                except Exception as audio_err:
                    logger.error(
                        f"[process_completed_tts_batch] Audio processing failed | "
                        f"result_id={result_id}, error={str(audio_err)}"
                    )
                    update_tts_result(
                        session=session,
                        result_id=result_record.id,
                        status=JobStatus.FAILED.value,
                        error_message=f"Audio processing failed: {str(audio_err)}",
                    )
                    failed_count += 1
            else:
                update_tts_result(
                    session=session,
                    result_id=result_record.id,
                    status=JobStatus.FAILED.value,
                    error_message=batch_result.get("error", "Unknown error"),
                )
                failed_count += 1

        session.commit()

    except Exception as e:
        logger.error(
            f"[process_completed_tts_batch] Failed to process batch results | "
            f"batch_job_id={batch_job.id}, error={str(e)}",
            exc_info=True,
        )
        raise

    logger.info(
        f"[process_completed_tts_batch] Batch results processed | "
        f"run_id={run.id}, provider={tts_provider}, "
        f"processed={processed_count}, failed={failed_count}"
    )


def _extract_audio_from_response(response: dict[str, Any]) -> str | None:
    """Extract base64-encoded audio data from a Gemini TTS response.

    Gemini TTS returns audio as base64-encoded PCM data in the
    inlineData field of the response parts.

    Args:
        response: Gemini response dictionary

    Returns:
        Base64 encoded audio string, or None if not found
    """
    # Navigate: candidates -> content -> parts -> inlineData -> data
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            inline_data = part.get("inlineData", {})
            if inline_data.get("data"):
                return inline_data["data"]
    return None
