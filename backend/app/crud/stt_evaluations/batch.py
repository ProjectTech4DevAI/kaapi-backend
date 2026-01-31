"""Batch submission functions for STT evaluation processing."""

import logging
from typing import Any

from sqlmodel import Session

from app.core.batch import GeminiBatchProvider
from app.core.cloud.storage import get_cloud_storage
from app.crud.stt_evaluations.result import update_stt_result
from app.crud.stt_evaluations.run import update_stt_run
from app.models import EvaluationRun
from app.models.stt_evaluation import STTResultStatus, STTSample
from app.services.stt_evaluations.gemini import GeminiClient, GeminiFilesManager

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


def _build_batch_requests(
    sample_file_mapping: list[tuple[int, int | None, str]],
    prompt: str = DEFAULT_TRANSCRIPTION_PROMPT,
) -> list[dict[str, Any]]:
    """Build JSONL batch request data from sample-file mappings.

    Each request follows the Gemini GenerateContentRequest format
    with a text prompt and file_data reference.

    Args:
        sample_file_mapping: List of (sample_id, result_id, google_file_uri) tuples
        prompt: Transcription prompt

    Returns:
        list[dict]: JSONL-compatible request dicts for GeminiBatchProvider
    """
    return [
        {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"file_data": {"file_uri": file_uri}},
                    ],
                    "role": "user",
                }
            ],
        }
        for _, _, file_uri in sample_file_mapping
    ]


def start_stt_evaluation_batch(
    *,
    session: Session,
    run: EvaluationRun,
    samples: list[STTSample],
    result_refs: list[dict[str, Any]],
    org_id: int,
    project_id: int,
) -> dict[str, Any]:
    """Upload audio files to Google and submit Gemini batch jobs.

    This function runs synchronously during the API request:
    1. Initializes GeminiClient
    2. Uploads audio files to Google Files API
    3. Builds batch requests
    4. Submits batch jobs per provider
    5. Stores batch_jobs and sample_file_mapping in run.score
    6. Updates run status to "processing"

    Args:
        session: Database session
        run: The evaluation run record
        samples: List of STT samples to process
        result_refs: List of result reference dicts with id, stt_sample_id, provider
        org_id: Organization ID
        project_id: Project ID

    Returns:
        dict: Result with batch job information

    Raises:
        Exception: If all batch submissions fail
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

    # Upload audio files to Google Files API
    files_manager = GeminiFilesManager(gemini_client.client)

    sample_file_mapping: list[tuple[int, int | None, str]] = []

    for sample in samples:
        try:
            # Get signed URL for S3 audio file
            signed_url = storage.get_signed_url(
                sample.object_store_url, expires_in=3600
            )

            # Extract filename from URL
            filename = sample.object_store_url.split("/")[-1]

            # Upload to Google Files API
            google_file_uri = files_manager.upload_from_url(
                signed_url=signed_url,
                filename=filename,
            )

            # Find the result record for this sample
            result_for_sample = next(
                (r for r in result_refs if r["stt_sample_id"] == sample.id),
                None,
            )

            sample_file_mapping.append(
                (
                    sample.id,
                    result_for_sample["id"] if result_for_sample else None,
                    google_file_uri,
                )
            )

            logger.info(
                f"[start_stt_evaluation_batch] Uploaded audio to Google | "
                f"sample_id: {sample.id}, file_uri: {google_file_uri}"
            )

        except Exception as e:
            logger.error(
                f"[start_stt_evaluation_batch] Failed to upload audio | "
                f"sample_id: {sample.id}, error: {str(e)}"
            )
            # Mark result as failed
            for ref in result_refs:
                if ref["stt_sample_id"] == sample.id:
                    update_stt_result(
                        session=session,
                        result_id=ref["id"],
                        status=STTResultStatus.FAILED.value,
                        error_message=f"Failed to upload audio: {str(e)}",
                    )

    if not sample_file_mapping:
        raise Exception("Failed to upload any audio files")

    # Build batch requests from uploaded files
    jsonl_data = _build_batch_requests(sample_file_mapping)

    # Process each provider using GeminiBatchProvider
    providers = run.providers or ["gemini-2.5-pro"]
    batch_jobs: dict[str, str] = {}

    for provider in providers:
        try:
            model = _get_model_for_provider(provider)
            batch_provider = GeminiBatchProvider(
                client=gemini_client.client, model=model
            )

            batch_result = batch_provider.create_batch(
                jsonl_data=jsonl_data,
                config={
                    "display_name": f"stt-eval-{run.id}-{provider}",
                    "model": model,
                },
            )

            batch_jobs[provider] = batch_result["provider_batch_id"]

            logger.info(
                f"[start_stt_evaluation_batch] Batch job submitted | "
                f"run_id: {run.id}, provider: {provider}, "
                f"batch_id: {batch_result['provider_batch_id']}"
            )

        except Exception as e:
            logger.error(
                f"[start_stt_evaluation_batch] Failed to submit batch | "
                f"provider: {provider}, error: {str(e)}"
            )
            # Update results for this provider as failed
            for ref in result_refs:
                if ref["provider"] == provider:
                    update_stt_result(
                        session=session,
                        result_id=ref["id"],
                        status=STTResultStatus.FAILED.value,
                        error_message=f"Batch submission failed: {str(e)}",
                    )

    if not batch_jobs:
        raise Exception("All batch submissions failed")

    # Store batch job info in run score for polling
    update_stt_run(
        session=session,
        run_id=run.id,
        status="processing",
        score={
            "batch_jobs": batch_jobs,
            "sample_file_mapping": [
                {"sample_id": s, "result_id": r, "file_uri": f}
                for s, r, f in sample_file_mapping
            ],
        },
    )

    logger.info(
        f"[start_stt_evaluation_batch] Batch submission complete | "
        f"run_id: {run.id}, batch_jobs: {list(batch_jobs.keys())}, "
        f"sample_count: {len(sample_file_mapping)}"
    )

    return {
        "success": True,
        "run_id": run.id,
        "batch_jobs": batch_jobs,
        "sample_count": len(sample_file_mapping),
    }
