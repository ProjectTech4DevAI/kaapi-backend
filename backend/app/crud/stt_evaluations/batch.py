"""Batch submission functions for STT evaluation processing."""

import logging
from typing import Any

from sqlmodel import Session

from app.core.batch import (
    GeminiBatchProvider,
    create_stt_batch_requests,
    start_batch_job,
)
from app.core.cloud.storage import get_cloud_storage
from app.crud.file import get_files_by_ids
from app.crud.stt_evaluations.result import update_stt_result
from app.crud.stt_evaluations.run import update_stt_run
from app.models import EvaluationRun
from app.models.stt_evaluation import STTResultStatus, STTSample
from app.services.stt_evaluations.gemini import GeminiClient

logger = logging.getLogger(__name__)

DEFAULT_TRANSCRIPTION_PROMPT = (
    "Generate a verbatim transcript of the speech in this audio file. "
    "Return only the transcription text without any formatting, timestamps, or metadata."
)

DEFAULT_MODEL = "gemini-2.5-pro"


def start_stt_evaluation_batch(
    *,
    session: Session,
    run: EvaluationRun,
    samples: list[STTSample],
    sample_to_result: dict[int, int],
    org_id: int,
    project_id: int,
    signed_url_expires_in: int = 86400,
) -> dict[str, Any]:
    """Generate signed URLs and submit Gemini batch job for STT evaluation.

    Args:
        session: Database session
        run: The evaluation run record
        samples: List of STT samples to process
        sample_to_result: Mapping of sample_id -> result_id for error handling
        org_id: Organization ID
        project_id: Project ID
        signed_url_expires_in: Signed URL expiry in seconds (default: 24 hours)

    Returns:
        dict: Result with batch job information

    Raises:
        Exception: If batch submission fails
    """
    logger.info(
        f"[start_stt_evaluation_batch] Starting batch submission | "
        f"run_id: {run.id}, sample_count: {len(samples)}"
    )

    # Initialize Gemini client
    gemini_client = GeminiClient.from_credentials(
        session=session,
        org_id=org_id,
        project_id=project_id,
    )

    # Get cloud storage for S3 access
    storage = get_cloud_storage(session=session, project_id=project_id)

    # Fetch file records to get object_store_url
    file_ids = [sample.file_id for sample in samples]
    file_records = get_files_by_ids(
        session=session,
        file_ids=file_ids,
        organization_id=org_id,
        project_id=project_id,
    )
    file_map = {f.id: f for f in file_records}

    # Generate signed URLs for audio files
    signed_urls: list[str] = []
    sample_keys: list[str] = []

    for sample in samples:
        try:
            # Get object_store_url from file record
            file_record = file_map.get(sample.file_id)
            if not file_record:
                raise ValueError(f"File record not found for file_id: {sample.file_id}")

            signed_url = storage.get_signed_url(
                file_record.object_store_url, expires_in=signed_url_expires_in
            )
            signed_urls.append(signed_url)
            sample_keys.append(str(sample.id))

        except Exception as e:
            logger.error(
                f"[start_stt_evaluation_batch] Failed to generate signed URL | "
                f"sample_id: {sample.id}, error: {str(e)}"
            )
            if sample.id in sample_to_result:
                update_stt_result(
                    session=session,
                    result_id=sample_to_result[sample.id],
                    status=STTResultStatus.FAILED.value,
                    error_message=f"Failed to generate signed URL: {str(e)}",
                )

    if not signed_urls:
        raise Exception("Failed to generate signed URLs for any audio files")

    jsonl_data = create_stt_batch_requests(
        signed_urls=signed_urls,
        prompt=DEFAULT_TRANSCRIPTION_PROMPT,
        keys=sample_keys,
    )

    model = (run.providers or [DEFAULT_MODEL])[0]
    model_path = f"models/{model}"

    batch_provider = GeminiBatchProvider(client=gemini_client.client, model=model_path)

    try:
        batch_job = start_batch_job(
            session=session,
            provider=batch_provider,
            provider_name="gemini",
            job_type="stt_evaluation",
            organization_id=org_id,
            project_id=project_id,
            jsonl_data=jsonl_data,
            config={"model": model},
        )

        logger.info(
            f"[start_stt_evaluation_batch] Batch job created | "
            f"run_id: {run.id}, batch_job_id: {batch_job.id}"
        )

    except Exception as e:
        logger.error(
            f"[start_stt_evaluation_batch] Failed to submit batch | "
            f"model: {model}, error: {str(e)}"
        )
        for result_id in sample_to_result.values():
            update_stt_result(
                session=session,
                result_id=result_id,
                status=STTResultStatus.FAILED.value,
                error_message=f"Batch submission failed: {str(e)}",
            )
        raise Exception(f"Batch submission failed: {str(e)}")

    # Link batch job to the evaluation run
    update_stt_run(
        session=session,
        run_id=run.id,
        status="processing",
        batch_job_id=batch_job.id,
    )

    logger.info(
        f"[start_stt_evaluation_batch] Batch submission complete | "
        f"run_id: {run.id}, batch_job_id: {batch_job.id}, "
        f"sample_count: {len(signed_urls)}"
    )

    return {
        "success": True,
        "run_id": run.id,
        "batch_job_id": batch_job.id,
        "provider_batch_id": batch_job.provider_batch_id,
        "sample_count": len(signed_urls),
    }
