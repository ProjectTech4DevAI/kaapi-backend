"""Celery task function for TTS evaluation result processing.

Processes completed Gemini TTS batch results: downloads JSONL,
extracts audio, converts PCM to WAV, uploads to S3, updates DB.
"""

import base64
import logging
import uuid
from typing import Any

from gevent import Timeout
from celery.exceptions import SoftTimeLimitExceeded
from sqlmodel import Session

from app.core.batch import BATCH_KEY, GeminiBatchProvider, GeminiClient
from app.core.cloud.storage import get_cloud_storage
from app.core.db import engine
from app.core.storage_utils import upload_to_object_store
from app.core.util import now
from app.crud.tts_evaluations.result import (
    get_pending_results_for_run,
    update_tts_result,
)
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.core.audio_utils import calculate_duration, pcm_to_wav
from app.services.tts_evaluations.constants import TTS_AUDIO_SUBDIRECTORY

logger = logging.getLogger(__name__)


def execute_tts_result_processing(
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance: Any,
    organization_id: int,
    evaluation_run_id: int,
    tts_provider: str,
    provider_batch_id: str,
    **kwargs: Any,
) -> dict:
    """Process completed TTS batch results in a Celery worker.

    Downloads batch results from Gemini, extracts audio, converts to WAV,
    uploads to S3, and updates TTSResult records.

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

            # Only PENDING rows, so reprocessing a batch skips rows already finished.
            pending = get_pending_results_for_run(
                session=session, run_id=evaluation_run_id, provider=tts_provider
            )
            result_map: dict[int, TTSResult] = {}
            for result in pending:
                result_map[result.id] = result

        logger.info(
            f"[execute_tts_result_processing] Pre-fetched result records | "
            f"run_id={evaluation_run_id}, provider={tts_provider}, "
            f"count={len(result_map)}"
        )

        # Download, conversion and uploads run with no session open; the detached
        # rows are modified in memory and written back in one short session below.
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

            result_record = result_map.get(result_id)

            if not result_record:
                logger.warning(
                    f"[execute_tts_result_processing] Result record not found or "
                    f"not pending | result_id={result_id}"
                )
                failed_count += 1
                continue

            if batch_result.get("response"):
                try:
                    audio_b64 = _extract_audio_from_response(batch_result["response"])

                    if not audio_b64:
                        result_record.status = JobStatus.FAILED.value
                        result_record.error_message = "No audio data in response"
                        result_record.updated_at = now()
                        failed_count += 1
                        continue

                    # Without validate=True, garbage decodes to b"" and is stored as a 0s "success".
                    pcm_data = base64.b64decode(audio_b64, validate=True)

                    wav_data = pcm_to_wav(pcm_data)
                    duration = calculate_duration(len(pcm_data))

                    audio_filename = f"{uuid.uuid4()}.wav"
                    audio_url = upload_to_object_store(
                        storage=storage,
                        content=wav_data,
                        filename=audio_filename,
                        subdirectory=TTS_AUDIO_SUBDIRECTORY,
                        content_type="audio/wav",
                    )

                    if not audio_url:
                        result_record.status = JobStatus.FAILED.value
                        result_record.error_message = (
                            "Audio upload to object store failed"
                        )
                        result_record.updated_at = now()
                        failed_count += 1
                        continue

                    result_record.object_store_url = audio_url
                    result_record.metadata_ = {
                        "duration_seconds": round(duration, 3),
                        "size_bytes": len(wav_data),
                    }
                    result_record.status = JobStatus.SUCCESS.value
                    result_record.updated_at = now()
                    processed_count += 1

                except Exception as audio_err:
                    logger.warning(
                        f"[execute_tts_result_processing] Audio processing failed | "
                        f"result_id={result_id}, error={str(audio_err)}"
                    )
                    result_record.status = JobStatus.FAILED.value
                    result_record.error_message = (
                        f"Audio processing failed: {str(audio_err)}"
                    )
                    result_record.updated_at = now()
                    failed_count += 1
            else:
                result_record.status = JobStatus.FAILED.value
                result_record.error_message = batch_result.get("error", "Unknown error")
                result_record.updated_at = now()
                failed_count += 1

        with Session(engine) as session:
            session.add_all(list(result_map.values()))
            session.commit()

        logger.info(
            f"[execute_tts_result_processing] Completed | "
            f"run_id={evaluation_run_id}, provider={tts_provider}, "
            f"processed={processed_count}, failed={failed_count}"
        )

        return {
            "success": True,
            "run_id": evaluation_run_id,
            "processed": processed_count,
            "failed": failed_count,
        }

    except (Timeout, SoftTimeLimitExceeded):
        logger.warning(
            f"[execute_tts_result_processing] TTS result processing timed out | run_id={evaluation_run_id}"
        )
        _fail_pending_results(
            run_id=evaluation_run_id,
            model=tts_provider,
            error_message="Result processing exceeded soft time limit",
        )
        raise

    except Exception as e:
        logger.error(
            f"[execute_tts_result_processing] Failed | "
            f"run_id={evaluation_run_id}, error={str(e)}",
            exc_info=True,
        )
        _fail_pending_results(
            run_id=evaluation_run_id,
            model=tts_provider,
            error_message=f"Result processing failed: {str(e)}",
        )
        return {"success": False, "error": str(e)}


def _fail_pending_results(*, run_id: int, model: str, error_message: str) -> None:
    """Fail only this model's leftover rows so the other models still decide the run."""
    with Session(engine) as session:
        pending = get_pending_results_for_run(
            session=session, run_id=run_id, provider=model
        )
        for result in pending:
            update_tts_result(
                session=session,
                result_id=result.id,
                status=JobStatus.FAILED.value,
                error_message=error_message,
            )
        session.commit()


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
