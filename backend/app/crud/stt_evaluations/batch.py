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
from app.crud.stt_evaluations.result import update_stt_result
from app.crud.stt_evaluations.run import update_stt_run
from app.models import EvaluationRun
from app.models.stt_evaluation import STTResultStatus, STTSample
from app.services.stt_evaluations.gemini import GeminiClient

logger = logging.getLogger(__name__)

# Default transcription prompt
DEFAULT_TRANSCRIPTION_PROMPT = (
    "Generate a verbatim transcript of the speech in this audio file. "
    "Return only the transcription text without any formatting, timestamps, or metadata."
)

# Provider name to Gemini model mapping
PROVIDER_MODEL_MAPPING: dict[str, str] = {
    "gemini-2.5-pro": "models/gemini-2.5-pro",
    "gemini-2.5-flash": "models/gemini-2.5-flash",
    "gemini-2.0-flash": "models/gemini-2.0-flash",
}


def _get_model_for_provider(provider: str) -> str:
    """Map provider name to Gemini model.

    Args:
        provider: Provider name

    Returns:
        str: Gemini model name
    """
    return PROVIDER_MODEL_MAPPING.get(provider, f"models/{provider}")


def start_stt_evaluation_batch(
    *,
    session: Session,
    run: EvaluationRun,
    samples: list[STTSample],
    result_refs: list[dict[str, Any]],
    org_id: int,
    project_id: int,
    signed_url_expires_in: int = 86400,
) -> dict[str, Any]:
    """Generate signed URLs and submit Gemini batch job for STT evaluation.

    This function runs synchronously during the API request:
    1. Initializes GeminiClient
    2. Generates signed URLs for audio files (valid for batch processing window)
    3. Builds batch requests using signed URLs directly
    4. Submits batch job via start_batch_job (creates BatchJob record)
    5. Links batch_job_id to the evaluation run
    6. Stores sample_file_mapping in run.score
    7. Updates run status to "processing"

    Args:
        session: Database session
        run: The evaluation run record
        samples: List of STT samples to process
        result_refs: List of result reference dicts with id, stt_sample_id, provider
        org_id: Organization ID
        project_id: Project ID
        signed_url_expires_in: Signed URL expiry in seconds (default: 24 hours for batch)

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

    # Generate signed URLs for audio files
    sample_url_mapping: list[dict[str, Any]] = []
    signed_urls: list[str] = []

    for sample in samples:
        try:
            # Get signed URL for S3 audio file
            # Use longer expiry for batch processing (up to 24 hours)
            signed_url = storage.get_signed_url(
                sample.object_store_url, expires_in=signed_url_expires_in
            )

            # Find the result record for this sample
            result_for_sample = next(
                (r for r in result_refs if r["stt_sample_id"] == sample.id),
                None,
            )

            sample_url_mapping.append(
                {
                    "sample_id": sample.id,
                    "result_id": result_for_sample["id"] if result_for_sample else None,
                    "signed_url": signed_url,
                }
            )
            signed_urls.append(signed_url)

            logger.info(
                f"[start_stt_evaluation_batch] Generated signed URL | "
                f"sample_id: {sample.id}"
            )

        except Exception as e:
            logger.error(
                f"[start_stt_evaluation_batch] Failed to generate signed URL | "
                f"sample_id: {sample.id}, error: {str(e)}"
            )
            # Mark result as failed
            for ref in result_refs:
                if ref["stt_sample_id"] == sample.id:
                    update_stt_result(
                        session=session,
                        result_id=ref["id"],
                        status=STTResultStatus.FAILED.value,
                        error_message=f"Failed to generate signed URL: {str(e)}",
                    )

    if not sample_url_mapping:
        raise Exception("Failed to generate signed URLs for any audio files")

    # Build batch requests using signed URLs directly (with mime type detection)
    jsonl_data = create_stt_batch_requests(
        signed_urls=signed_urls,
        prompt=DEFAULT_TRANSCRIPTION_PROMPT,
    )

    # Use first provider (STT evaluations use one provider per run)
    providers = run.providers or ["gemini-2.5-pro"]
    provider = providers[0]
    model = _get_model_for_provider(provider)

    # Create batch job using the standard batch operations
    batch_provider = GeminiBatchProvider(client=gemini_client.client, model=model)

    try:
        batch_job = start_batch_job(
            session=session,
            provider=batch_provider,
            provider_name="gemini",
            job_type="stt_evaluation",
            organization_id=org_id,
            project_id=project_id,
            jsonl_data=jsonl_data,
            config={
                "display_name": f"stt-eval-{run.id}-{provider}",
                "model": model,
                "stt_provider": provider,
            },
        )

        logger.info(
            f"[start_stt_evaluation_batch] Batch job created | "
            f"run_id: {run.id}, batch_job_id: {batch_job.id}, "
            f"provider_batch_id: {batch_job.provider_batch_id}"
        )

    except Exception as e:
        logger.error(
            f"[start_stt_evaluation_batch] Failed to submit batch | "
            f"provider: {provider}, error: {str(e)}"
        )
        # Update all results as failed
        for ref in result_refs:
            update_stt_result(
                session=session,
                result_id=ref["id"],
                status=STTResultStatus.FAILED.value,
                error_message=f"Batch submission failed: {str(e)}",
            )
        raise Exception(f"Batch submission failed: {str(e)}")

    # Link batch job to the evaluation run and store sample mapping
    update_stt_run(
        session=session,
        run_id=run.id,
        status="processing",
        batch_job_id=batch_job.id,
        score={
            "sample_file_mapping": sample_url_mapping,
        },
    )

    logger.info(
        f"[start_stt_evaluation_batch] Batch submission complete | "
        f"run_id: {run.id}, batch_job_id: {batch_job.id}, "
        f"sample_count: {len(sample_url_mapping)}"
    )

    return {
        "success": True,
        "run_id": run.id,
        "batch_job_id": batch_job.id,
        "provider_batch_id": batch_job.provider_batch_id,
        "sample_count": len(sample_url_mapping),
    }
