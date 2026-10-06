"""Celery task function for TTS evaluation result processing.

Processes completed Gemini TTS batch results: downloads JSONL,
extracts audio, converts PCM to WAV, uploads to S3, updates DB.

The DB session is only opened for short prefetch/write/finalize steps; the
download, audio conversion and S3 uploads run with no connection checked out.
"""

import base64
import logging
import uuid
from typing import Any

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session

from app.core.audio_utils import calculate_duration, pcm_to_wav
from app.core.batch import BATCH_KEY, GeminiBatchProvider, GeminiClient
from app.core.cloud.storage import CloudStorage, get_cloud_storage
from app.core.db import engine
from app.core.storage_utils import upload_to_object_store
from app.crud.tts_evaluations.result import (
    bulk_update_pending_tts_results,
    get_pending_results_for_run,
)
from app.crud.tts_evaluations.run import finalize_tts_run_status, update_tts_run
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResultUpdate
from app.services.tts_evaluations.constants import (
    TTS_AUDIO_CONTENT_TYPE,
    TTS_AUDIO_FILE_EXTENSION,
    TTS_AUDIO_SUBDIRECTORY,
    TTS_RESULT_WRITE_CHUNK_SIZE,
)

logger = logging.getLogger(__name__)

DURATION_DECIMALS = 3


def execute_tts_result_processing(
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance: Task,
    organization_id: int,
    evaluation_run_id: int,
    tts_provider: str,
    provider_batch_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Process completed TTS batch results in a Celery worker.

    Downloads batch results from Gemini, extracts audio, converts to WAV,
    uploads to S3, and updates TTSResult records. Only results still PENDING
    are processed, so reprocessing a batch skips rows already finished.

    Args:
        project_id: Project ID
        job_id: Batch job ID (as string)
        task_id: Celery task ID
        task_instance: Celery task instance
        organization_id: Organization ID
        evaluation_run_id: Evaluation run ID
        tts_provider: TTS provider/model name
        provider_batch_id: Gemini batch job ID

    Returns:
        dict: Result summary with processed/failed counts
    """
    logger.info(
        f"[execute_tts_result_processing] Starting | "
        f"run_id={evaluation_run_id}, batch_job_id={job_id}, "
        f"provider={tts_provider}, celery_task_id={task_id}"
    )

    try:
        with Session(engine) as session:
            gemini_client = GeminiClient.from_credentials(
                session=session,
                org_id=organization_id,
                project_id=project_id,
            )
            storage = get_cloud_storage(session=session, project_id=project_id)
            pending_rows = get_pending_results_for_run(
                session=session, run_id=evaluation_run_id, provider=tts_provider
            )
            pending_ids: set[int] = set()
            for row in pending_rows:
                pending_ids.add(row.id)

        logger.info(
            f"[execute_tts_result_processing] Pre-fetched pending result ids | "
            f"run_id={evaluation_run_id}, provider={tts_provider}, "
            f"count={len(pending_ids)}"
        )

        batch_provider = GeminiBatchProvider(client=gemini_client.client)
        results = batch_provider.download_batch_results(provider_batch_id)

        logger.info(
            f"[execute_tts_result_processing] Got batch results | "
            f"run_id={evaluation_run_id}, result_count={len(results)}"
        )

        if results:
            first = results[0]
            resp = first.get("response") or {}
            resp_keys = (
                list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__
            )
            logger.info(
                f"[execute_tts_result_processing] First result structure | "
                f"keys={list(first.keys())}, response_keys={resp_keys}"
            )

        processed_count = 0
        failed_count = 0
        skipped_count = 0
        pending_updates: list[TTSResultUpdate] = []

        for batch_result in results:
            custom_id = batch_result[BATCH_KEY]
            try:
                result_id = int(custom_id)
            except (ValueError, TypeError):
                logger.warning(
                    f"[execute_tts_result_processing] Invalid {BATCH_KEY} | "
                    f"run_id={evaluation_run_id}, {BATCH_KEY}={custom_id}"
                )
                failed_count += 1
                continue

            if result_id not in pending_ids:
                logger.info(
                    f"[execute_tts_result_processing] Result not pending, skipping | "
                    f"result_id={result_id}"
                )
                skipped_count += 1
                continue

            update = _build_result_update(
                result_id=result_id, batch_result=batch_result, storage=storage
            )
            if update.status == JobStatus.SUCCESS:
                processed_count += 1
            else:
                failed_count += 1

            pending_updates.append(update)
            if len(pending_updates) >= TTS_RESULT_WRITE_CHUNK_SIZE:
                _write_result_updates(pending_updates)
                pending_updates = []

        _write_result_updates(pending_updates)

        with Session(engine) as session:
            run = finalize_tts_run_status(session=session, run_id=evaluation_run_id)
            final_status = run.status if run else None

        logger.info(
            f"[execute_tts_result_processing] Completed | "
            f"run_id={evaluation_run_id}, provider={tts_provider}, "
            f"processed={processed_count}, failed={failed_count}, "
            f"skipped={skipped_count}, run_status={final_status}"
        )

        return {
            "success": True,
            "run_id": evaluation_run_id,
            "processed": processed_count,
            "failed": failed_count,
            "run_status": final_status,
        }

    except (Timeout, SoftTimeLimitExceeded):
        timeout_err = TimeoutError("Task exceeded soft time limit")
        logger.warning(
            f"[execute_tts_result_processing] TTS result processing timed out | run_id={evaluation_run_id}"
        )
        _mark_run_failed(run_id=evaluation_run_id, error_message=str(timeout_err))
        raise

    except Exception as e:
        logger.error(
            f"[execute_tts_result_processing] Failed | "
            f"run_id={evaluation_run_id}, error={str(e)}",
            exc_info=True,
        )
        _mark_run_failed(
            run_id=evaluation_run_id,
            error_message=f"Result processing failed: {str(e)}",
        )
        return {"success": False, "error": str(e)}


def _build_result_update(
    *,
    result_id: int,
    batch_result: dict[str, Any],
    storage: CloudStorage,
) -> TTSResultUpdate:
    """Convert one Gemini batch result into a terminal update (uploads the WAV)."""
    if not batch_result.get("response"):
        return TTSResultUpdate(
            result_id=result_id,
            status=JobStatus.FAILED,
            error_message=batch_result.get("error", "Unknown error"),
        )

    try:
        audio_b64 = _extract_audio_from_response(batch_result["response"])

        if not audio_b64:
            return TTSResultUpdate(
                result_id=result_id,
                status=JobStatus.FAILED,
                error_message="No audio data in response",
            )

        # Without validate=True, garbage input decodes to b"" and would be stored as a 0s "success".
        pcm_data = base64.b64decode(audio_b64, validate=True)
        wav_data = pcm_to_wav(pcm_data)
        duration = calculate_duration(len(pcm_data))

        audio_url = upload_to_object_store(
            storage=storage,
            content=wav_data,
            filename=f"{uuid.uuid4()}.{TTS_AUDIO_FILE_EXTENSION}",
            subdirectory=TTS_AUDIO_SUBDIRECTORY,
            content_type=TTS_AUDIO_CONTENT_TYPE,
        )

        if not audio_url:
            return TTSResultUpdate(
                result_id=result_id,
                status=JobStatus.FAILED,
                error_message="Audio upload to object store failed",
            )

        return TTSResultUpdate(
            result_id=result_id,
            status=JobStatus.SUCCESS,
            object_store_url=audio_url,
            metadata={
                "duration_seconds": round(duration, DURATION_DECIMALS),
                "size_bytes": len(wav_data),
            },
        )

    except Exception as audio_err:
        logger.warning(
            f"[_build_result_update] Audio processing failed | "
            f"result_id={result_id}, error={str(audio_err)}"
        )
        return TTSResultUpdate(
            result_id=result_id,
            status=JobStatus.FAILED,
            error_message=f"Audio processing failed: {str(audio_err)}",
        )


def _write_result_updates(updates: list[TTSResultUpdate]) -> None:
    if not updates:
        return
    with Session(engine) as session:
        bulk_update_pending_tts_results(session=session, updates=updates)


def _mark_run_failed(*, run_id: int, error_message: str) -> None:
    with Session(engine) as session:
        update_tts_run(
            session=session,
            run_id=run_id,
            status="failed",
            error_message=error_message,
        )


def _extract_audio_from_response(response: dict[str, Any]) -> str | None:
    """Extract base64-encoded audio data from a Gemini TTS response.

    Gemini TTS returns audio as base64-encoded PCM data in the
    inlineData field of the response parts. Handles both camelCase
    (REST API) and snake_case (Python SDK / batch JSONL) field names.

    Args:
        response: Gemini response dictionary

    Returns:
        Base64 encoded audio string, or None if not found
    """
    # Navigate: candidates -> content -> parts -> inlineData/inline_data -> data
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            # Handle both camelCase (inlineData) and snake_case (inline_data)
            inline_data = part.get("inlineData") or part.get("inline_data") or {}
            if inline_data.get("data"):
                return inline_data["data"]

    part_keys = [
        list(p.keys())
        for c in response.get("candidates", [])
        for p in c.get("content", {}).get("parts", [])
    ]
    logger.warning(
        f"[_extract_audio_from_response] No audio data found | "
        f"response_keys={list(response.keys())}, parts={part_keys}"
    )
    return None
