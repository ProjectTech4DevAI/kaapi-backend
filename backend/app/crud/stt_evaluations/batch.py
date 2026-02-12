"""Batch submission functions for STT evaluation processing."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from sqlmodel import Session

from app.core.batch import (
    GeminiBatchProvider,
    create_stt_batch_requests,
    start_batch_job,
)
from app.core.cloud.storage import get_cloud_storage
from app.crud.file import get_files_by_ids
from app.crud.stt_evaluations.run import update_stt_run
from app.models import EvaluationRun
from app.models.stt_evaluation import STTSample
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
    org_id: int,
    project_id: int,
    signed_url_expires_in: int = 86400,
) -> dict[str, Any]:
    """Generate signed URLs and submit Gemini batch jobs for STT evaluation.

    Submits one batch job per model. Each batch job is tracked via
    its config containing evaluation_run_id and stt_provider.

    Args:
        session: Database session
        run: The evaluation run record
        samples: List of STT samples to process
        org_id: Organization ID
        project_id: Project ID
        signed_url_expires_in: Signed URL expiry in seconds (default: 24 hours)

    Returns:
        dict: Result with batch job information per model

    Raises:
        Exception: If batch submission fails for all models
    """
    models = run.providers or [DEFAULT_MODEL]

    logger.info(
        f"[start_stt_evaluation_batch] Starting batch submission | "
        f"run_id: {run.id}, sample_count: {len(samples)}, "
        f"models: {models}"
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

    # Generate signed URLs for audio files concurrently (shared across all models)
    signed_urls: list[str] = []
    sample_keys: list[str] = []
    failed_samples: list[tuple[STTSample, str]] = []

    def _generate_signed_url(
        sample: STTSample,
    ) -> tuple[STTSample, str | None, str | None]:
        """Generate a signed URL for a single sample. Thread-safe."""
        file_record = file_map.get(sample.file_id)
        if not file_record:
            return sample, None, f"File record not found for file_id: {sample.file_id}"
        try:
            url = storage.get_signed_url(
                file_record.object_store_url, expires_in=signed_url_expires_in
            )
            return sample, url, None
        except Exception as e:
            return sample, None, str(e)

    with ThreadPoolExecutor(max_workers=10) as executor:
        sign_url_tasks = {
            executor.submit(_generate_signed_url, sample): sample for sample in samples
        }

        for completed_task in as_completed(sign_url_tasks):
            sample, url, error = completed_task.result()
            if url:
                signed_urls.append(url)
                sample_keys.append(str(sample.id))
            else:
                failed_samples.append((sample, error))
                logger.error(
                    f"[start_stt_evaluation_batch] Failed to generate signed URL | "
                    f"sample_id: {sample.id}, error: {error}"
                )

    if failed_samples:
        logger.warning(
            f"[start_stt_evaluation_batch] Signed URL failures | "
            f"run_id: {run.id}, failed_count: {len(failed_samples)}, "
            f"succeeded_count: {len(signed_urls)}"
        )

    if not signed_urls:
        raise Exception("Failed to generate signed URLs for any audio files")

    # Create JSONL batch requests (shared across all models)
    jsonl_data = create_stt_batch_requests(
        signed_urls=signed_urls,
        prompt=DEFAULT_TRANSCRIPTION_PROMPT,
        keys=sample_keys,
    )

    # Submit one batch job per model
    batch_jobs: dict[str, Any] = {}
    first_batch_job_id: int | None = None

    for model in models:
        model_path = f"models/{model}"
        batch_provider = GeminiBatchProvider(
            client=gemini_client.client, model=model_path
        )

        try:
            batch_job = start_batch_job(
                session=session,
                provider=batch_provider,
                provider_name="google",
                job_type="stt_evaluation",
                organization_id=org_id,
                project_id=project_id,
                jsonl_data=jsonl_data,
                config={
                    "model": model,
                    "stt_provider": model,
                    "evaluation_run_id": run.id,
                },
            )

            batch_jobs[model] = {
                "batch_job_id": batch_job.id,
                "provider_batch_id": batch_job.provider_batch_id,
            }

            if first_batch_job_id is None:
                first_batch_job_id = batch_job.id

            logger.info(
                f"[start_stt_evaluation_batch] Batch job created | "
                f"run_id: {run.id}, model: {model}, "
                f"batch_job_id: {batch_job.id}"
            )

        except Exception as e:
            logger.error(
                f"[start_stt_evaluation_batch] Failed to submit batch | "
                f"model: {model}, error: {str(e)}"
            )

    if not batch_jobs:
        raise Exception("Batch submission failed for all models")

    # Link first batch job to the evaluation run (for pending run detection)
    update_stt_run(
        session=session,
        run_id=run.id,
        status="processing",
        batch_job_id=first_batch_job_id,
    )

    logger.info(
        f"[start_stt_evaluation_batch] Batch submission complete | "
        f"run_id: {run.id}, models_submitted: {list(batch_jobs.keys())}, "
        f"sample_count: {len(signed_urls)}"
    )

    return {
        "success": True,
        "run_id": run.id,
        "batch_jobs": batch_jobs,
        "sample_count": len(signed_urls),
    }
