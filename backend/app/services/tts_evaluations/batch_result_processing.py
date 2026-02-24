"""Celery task function for TTS evaluation result processing.

Processes completed Gemini TTS batch results: downloads JSONL,
extracts audio, converts PCM to WAV, uploads to S3, updates DB.
"""

import base64
import logging
import uuid
from typing import Any

from sqlmodel import Session, select

from app.core.batch import BATCH_KEY, GeminiBatchProvider
from app.core.cloud.storage import get_cloud_storage
from app.core.db import engine
from app.core.storage_utils import upload_to_object_store
from app.crud.tts_evaluations.result import (
    count_results_by_status,
    update_tts_result,
)
from app.crud.tts_evaluations.run import update_tts_run
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult
from app.services.stt_evaluations.gemini import GeminiClient
from app.services.tts_evaluations.audio import calculate_duration, pcm_to_wav

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

    with Session(engine) as session:
        try:
            # Initialize Gemini client and batch provider
            gemini_client = GeminiClient.from_credentials(
                session=session,
                org_id=organization_id,
                project_id=project_id,
            )
            batch_provider = GeminiBatchProvider(client=gemini_client.client)

            # Get cloud storage for S3 uploads
            storage = get_cloud_storage(session=session, project_id=project_id)

            # Download batch results
            results = batch_provider.download_batch_results(provider_batch_id)

            logger.info(
                f"[execute_tts_result_processing] Got batch results | "
                f"run_id={evaluation_run_id}, result_count={len(results)}"
            )

            if results:
                first = results[0]
                resp = first.get("response") or {}
                logger.info(
                    f"[execute_tts_result_processing] First result structure | "
                    f"keys={list(first.keys())}, "
                    f"response_keys={list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__}"
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

                # Find result record
                stmt = select(TTSResult).where(
                    TTSResult.id == result_id,
                    TTSResult.evaluation_run_id == evaluation_run_id,
                    TTSResult.provider == tts_provider,
                )
                result_record = session.exec(stmt).one_or_none()

                if not result_record:
                    logger.warning(
                        f"[execute_tts_result_processing] Result record not found | "
                        f"result_id={result_id}"
                    )
                    failed_count += 1
                    continue

                if batch_result.get("response"):
                    try:
                        audio_b64 = _extract_audio_from_response(
                            batch_result["response"]
                        )

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
                            subdirectory="evaluations/tts/audio",
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
                            f"[execute_tts_result_processing] Audio processing failed | "
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

            # Finalize run status
            status_counts = count_results_by_status(
                session=session, run_id=evaluation_run_id
            )
            pending = status_counts.get(JobStatus.PENDING.value, 0)
            total_failed = status_counts.get(JobStatus.FAILED.value, 0)

            final_status = "completed" if pending == 0 else "processing"
            error_message = (
                f"{total_failed} synthesis(es) failed" if total_failed > 0 else None
            )

            update_tts_run(
                session=session,
                run_id=evaluation_run_id,
                status=final_status,
                error_message=error_message,
            )

            logger.info(
                f"[execute_tts_result_processing] Completed | "
                f"run_id={evaluation_run_id}, provider={tts_provider}, "
                f"processed={processed_count}, failed={failed_count}, "
                f"run_status={final_status}"
            )

            return {
                "success": True,
                "run_id": evaluation_run_id,
                "processed": processed_count,
                "failed": failed_count,
                "run_status": final_status,
            }

        except Exception as e:
            logger.error(
                f"[execute_tts_result_processing] Failed | "
                f"run_id={evaluation_run_id}, error={str(e)}",
                exc_info=True,
            )
            update_tts_run(
                session=session,
                run_id=evaluation_run_id,
                status="failed",
                error_message=f"Result processing failed: {str(e)}",
            )
            return {"success": False, "error": str(e)}


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

    logger.warning(
        f"[_extract_audio_from_response] No audio data found | "
        f"response_keys={list(response.keys())}, "
        f"parts={[list(p.keys()) for c in response.get('candidates', []) for p in c.get('content', {}).get('parts', [])]}"
    )
    return None
