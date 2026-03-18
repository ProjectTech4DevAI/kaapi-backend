"""Batch submission functions for STT evaluation processing."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, NamedTuple

from sqlmodel import Session

from app.core.batch import (
    GeminiBatchProvider,
    GeminiClient,
    create_stt_batch_requests,
    start_batch_job,
)
from app.core.cloud.storage import get_cloud_storage
from app.core.storage_utils import get_mime_from_url
from app.crud.file import get_files_by_ids
from app.crud.stt_evaluations.run import update_stt_run
from app.models import EvaluationRun
from app.models.batch_job import BatchJobType
from app.models.stt_evaluation import STTSample

logger = logging.getLogger(__name__)


class _UploadResult(NamedTuple):
    sample: STTSample
    file_uri: str | None
    file_name: str | None
    mime_type: str | None
    error: str | None


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
) -> dict[str, Any]:
    """Upload audio files to Gemini and submit batch jobs for STT evaluation.

    Downloads audio from S3 and uploads to Gemini File API, then submits
    one batch job per model. Each batch job is tracked via its config
    containing evaluation_run_id and stt_provider.

    Args:
        session: Database session
        run: The evaluation run record
        samples: List of STT samples to process
        org_id: Organization ID
        project_id: Project ID

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

    # Upload audio files to Gemini File API concurrently (shared across all models)
    upload_provider = GeminiBatchProvider(client=gemini_client.client)

    file_uris: list[str] = []
    mime_types: list[str] = []
    sample_keys: list[str] = []
    gemini_file_names: list[str] = []
    failed_samples: list[tuple[STTSample, str]] = []

    def _upload_to_gemini(sample: STTSample) -> _UploadResult:
        """Download from S3 and upload to Gemini File API. Thread-safe."""
        file_record = file_map.get(sample.file_id)
        if not file_record:
            return _UploadResult(
                sample=sample,
                file_uri=None,
                file_name=None,
                mime_type=None,
                error=f"File record not found for file_id: {sample.file_id}",
            )
        try:
            # Detect MIME type from S3 URL path
            mime_type = get_mime_from_url(file_record.object_store_url)
            if mime_type is None:
                mime_type = file_record.content_type or "audio/mpeg"

            # Download audio from S3
            body = storage.stream(file_record.object_store_url)
            audio_bytes = body.read()

            # Upload to Gemini File API
            file_name, file_uri = upload_provider.upload_audio_file(
                content=audio_bytes,
                mime_type=mime_type,
                display_name=f"stt-eval-{run.id}-sample-{sample.id}",
            )
            return _UploadResult(
                sample=sample,
                file_uri=file_uri,
                file_name=file_name,
                mime_type=mime_type,
                error=None,
            )
        except Exception as e:
            return _UploadResult(
                sample=sample,
                file_uri=None,
                file_name=None,
                mime_type=None,
                error=str(e),
            )

    with ThreadPoolExecutor(max_workers=5) as executor:
        upload_tasks = {
            executor.submit(_upload_to_gemini, sample): sample for sample in samples
        }

        for completed_task in as_completed(upload_tasks):
            result = completed_task.result()
            if result.file_uri:
                file_uris.append(result.file_uri)
                mime_types.append(result.mime_type)
                sample_keys.append(str(result.sample.id))
                gemini_file_names.append(result.file_name)
            else:
                failed_samples.append((result.sample, result.error))
                logger.error(
                    f"[start_stt_evaluation_batch] Failed to upload to Gemini | "
                    f"sample_id: {result.sample.id}, error: {result.error}"
                )

    if failed_samples:
        logger.warning(
            f"[start_stt_evaluation_batch] Gemini upload failures | "
            f"run_id: {run.id}, failed_count: {len(failed_samples)}, "
            f"succeeded_count: {len(file_uris)}"
        )

    if not file_uris:
        raise Exception("Failed to upload audio files to Gemini for any samples")

    # Create JSONL batch requests (shared across all models)
    jsonl_data = create_stt_batch_requests(
        file_uris=file_uris,
        mime_types=mime_types,
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
                job_type=BatchJobType.STT_EVALUATION,
                organization_id=org_id,
                project_id=project_id,
                jsonl_data=jsonl_data,
                config={
                    "model": model,
                    "stt_provider": model,
                    "evaluation_run_id": run.id,
                    "gemini_audio_file_ids": gemini_file_names,
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
        f"sample_count: {len(file_uris)}"
    )

    return {
        "success": True,
        "run_id": run.id,
        "batch_jobs": batch_jobs,
        "sample_count": len(file_uris),
    }
