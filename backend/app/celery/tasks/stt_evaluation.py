"""Celery tasks for STT evaluation processing."""

import logging
from typing import Any

from asgi_correlation_id import correlation_id
from celery import current_task
from sqlmodel import Session, select

from app.celery.celery_app import celery_app
from app.core.batch import GeminiBatchProvider
from app.core.batch.gemini import BatchJobState
from app.core.db import engine
from app.core.cloud.storage import get_cloud_storage
from app.crud.stt_evaluations import (
    get_stt_run_by_id,
    get_samples_by_dataset_id,
    get_stt_dataset_by_id,
    update_stt_run,
    create_stt_results,
    update_stt_result,
    count_results_by_status,
)
from app.models.stt_evaluation import STTResult, STTResultStatus
from app.services.stt_evaluations.gemini import (
    GeminiClient,
    GeminiFilesManager,
)

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

# Maximum number of polls (24 hours with 30s intervals)
MAX_POLL_COUNT = 2880


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


@celery_app.task(bind=True, queue="low_priority")
def process_stt_evaluation(
    self,
    evaluation_run_id: int,
    org_id: int,
    project_id: int,
    trace_id: str,
    **kwargs,
) -> dict[str, Any]:
    """Process an STT evaluation run.

    This task:
    1. Fetches the evaluation run and dataset samples
    2. Uploads audio files to Google Files API
    3. Creates batch requests via GeminiBatchProvider
    4. Submits batch jobs per provider
    5. Schedules polling task

    Args:
        evaluation_run_id: ID of the evaluation run
        org_id: Organization ID
        project_id: Project ID
        trace_id: Correlation/trace ID

    Returns:
        dict: Result with batch job information
    """
    task_id = current_task.request.id
    correlation_id.set(trace_id)

    logger.info(
        f"[process_stt_evaluation] Starting STT evaluation | "
        f"run_id: {evaluation_run_id}, task_id: {task_id}"
    )

    try:
        with Session(engine) as session:
            # Get the evaluation run
            run = get_stt_run_by_id(
                session=session,
                run_id=evaluation_run_id,
                org_id=org_id,
                project_id=project_id,
            )

            if not run:
                logger.error(
                    f"[process_stt_evaluation] Run not found | run_id: {evaluation_run_id}"
                )
                return {"success": False, "error": "Evaluation run not found"}

            # Get the dataset
            dataset = get_stt_dataset_by_id(
                session=session,
                dataset_id=run.dataset_id,
                org_id=org_id,
                project_id=project_id,
            )

            if not dataset:
                update_stt_run(
                    session=session,
                    run_id=evaluation_run_id,
                    status="failed",
                    error_message="Dataset not found",
                )
                return {"success": False, "error": "Dataset not found"}

            # Get all samples
            samples, total = get_samples_by_dataset_id(
                session=session,
                dataset_id=run.dataset_id,
                org_id=org_id,
                project_id=project_id,
                limit=10000,  # Get all samples
            )

            if not samples:
                update_stt_run(
                    session=session,
                    run_id=evaluation_run_id,
                    status="failed",
                    error_message="No samples in dataset",
                )
                return {"success": False, "error": "No samples in dataset"}

            # Update run with total items
            providers = run.providers or ["gemini-2.5-pro"]
            total_items = len(samples) * len(providers)

            update_stt_run(
                session=session,
                run_id=evaluation_run_id,
                status="processing",
            )

            # Create result records for each sample and provider
            results = create_stt_results(
                session=session,
                samples=samples,
                evaluation_run_id=evaluation_run_id,
                org_id=org_id,
                project_id=project_id,
                providers=providers,
            )

            # Extract result data before session closes to avoid DetachedInstanceError
            result_refs = [
                {"id": r.id, "stt_sample_id": r.stt_sample_id, "provider": r.provider}
                for r in results
            ]

            # Update total items
            run.total_items = total_items
            session.add(run)
            session.commit()

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

            sample_file_mapping = []  # [(sample_id, result_id, google_file_uri)]

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
                        f"[process_stt_evaluation] Uploaded audio to Google | "
                        f"sample_id: {sample.id}, file_uri: {google_file_uri}"
                    )

                except Exception as e:
                    logger.error(
                        f"[process_stt_evaluation] Failed to upload audio | "
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

        # Build batch requests from uploaded files
        jsonl_data = _build_batch_requests(sample_file_mapping)

        # Process each provider using GeminiBatchProvider
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
                        "display_name": f"stt-eval-{evaluation_run_id}-{provider}",
                        "model": model,
                    },
                )

                batch_jobs[provider] = batch_result["provider_batch_id"]

                logger.info(
                    f"[process_stt_evaluation] Batch job submitted | "
                    f"run_id: {evaluation_run_id}, provider: {provider}, "
                    f"batch_id: {batch_result['provider_batch_id']}"
                )

            except Exception as e:
                logger.error(
                    f"[process_stt_evaluation] Failed to submit batch | "
                    f"provider: {provider}, error: {str(e)}"
                )
                # Update results for this provider as failed
                with Session(engine) as session:
                    for ref in result_refs:
                        if ref["provider"] == provider:
                            update_stt_result(
                                session=session,
                                result_id=ref["id"],
                                status=STTResultStatus.FAILED.value,
                                error_message=f"Batch submission failed: {str(e)}",
                            )

        if not batch_jobs:
            with Session(engine) as session:
                update_stt_run(
                    session=session,
                    run_id=evaluation_run_id,
                    status="failed",
                    error_message="All batch submissions failed",
                )
            return {"success": False, "error": "All batch submissions failed"}

        # Store batch job info in run score for polling
        with Session(engine) as session:
            update_stt_run(
                session=session,
                run_id=evaluation_run_id,
                score={
                    "batch_jobs": batch_jobs,
                    "sample_file_mapping": [
                        {"sample_id": s, "result_id": r, "file_uri": f}
                        for s, r, f in sample_file_mapping
                    ],
                },
            )

        # Schedule polling task
        poll_stt_batch_status.apply_async(
            kwargs={
                "evaluation_run_id": evaluation_run_id,
                "org_id": org_id,
                "project_id": project_id,
                "trace_id": trace_id,
                "batch_jobs": batch_jobs,
            },
            countdown=30,  # Wait 30 seconds before first poll
        )

        return {
            "success": True,
            "run_id": evaluation_run_id,
            "batch_jobs": batch_jobs,
            "sample_count": len(samples),
        }

    except Exception as e:
        logger.error(
            f"[process_stt_evaluation] Failed to process evaluation | "
            f"run_id: {evaluation_run_id}, error: {str(e)}",
            exc_info=True,
        )

        with Session(engine) as session:
            update_stt_run(
                session=session,
                run_id=evaluation_run_id,
                status="failed",
                error_message=str(e),
            )

        return {"success": False, "error": str(e)}


@celery_app.task(bind=True, queue="low_priority")
def poll_stt_batch_status(
    self,
    evaluation_run_id: int,
    org_id: int,
    project_id: int,
    trace_id: str,
    batch_jobs: dict[str, str],
    poll_count: int = 0,
    **kwargs,
) -> dict[str, Any]:
    """Poll Gemini batch job status using GeminiBatchProvider.

    Args:
        evaluation_run_id: ID of the evaluation run
        org_id: Organization ID
        project_id: Project ID
        trace_id: Correlation/trace ID
        batch_jobs: Dict of provider -> batch_id
        poll_count: Number of times we've polled

    Returns:
        dict: Status information
    """
    correlation_id.set(trace_id)

    if poll_count >= MAX_POLL_COUNT:
        logger.error(
            f"[poll_stt_batch_status] Polling timed out | "
            f"run_id: {evaluation_run_id}"
        )
        with Session(engine) as session:
            update_stt_run(
                session=session,
                run_id=evaluation_run_id,
                status="failed",
                error_message="Batch processing timed out after 24 hours",
            )
        return {"success": False, "error": "Timeout"}

    logger.info(
        f"[poll_stt_batch_status] Polling batch status | "
        f"run_id: {evaluation_run_id}, poll_count: {poll_count}"
    )

    try:
        with Session(engine) as session:
            # Initialize Gemini client and batch provider
            gemini_client = GeminiClient.from_credentials(
                session=session,
                org_id=org_id,
                project_id=project_id,
            )
            batch_provider = GeminiBatchProvider(client=gemini_client.client)

            all_complete = True
            any_success = False

            for provider, batch_id in batch_jobs.items():
                status = batch_provider.get_batch_status(batch_id)
                provider_status = status["provider_status"]

                logger.info(
                    f"[poll_stt_batch_status] Batch status | "
                    f"provider: {provider}, batch_id: {batch_id}, "
                    f"state: {provider_status}"
                )

                is_terminal = provider_status in {
                    BatchJobState.SUCCEEDED.value,
                    BatchJobState.FAILED.value,
                    BatchJobState.CANCELLED.value,
                    BatchJobState.EXPIRED.value,
                }

                if not is_terminal:
                    all_complete = False
                elif provider_status == BatchJobState.SUCCEEDED.value:
                    any_success = True

            if not all_complete:
                # Re-schedule polling
                poll_stt_batch_status.apply_async(
                    kwargs={
                        "evaluation_run_id": evaluation_run_id,
                        "org_id": org_id,
                        "project_id": project_id,
                        "trace_id": trace_id,
                        "batch_jobs": batch_jobs,
                        "poll_count": poll_count + 1,
                    },
                    countdown=30,
                )
                return {"success": True, "status": "polling", "poll_count": poll_count}

            # All batches complete - process results
            if any_success:
                process_stt_batch_results.apply_async(
                    kwargs={
                        "evaluation_run_id": evaluation_run_id,
                        "org_id": org_id,
                        "project_id": project_id,
                        "trace_id": trace_id,
                        "batch_jobs": batch_jobs,
                    },
                )
                return {"success": True, "status": "processing_results"}
            else:
                update_stt_run(
                    session=session,
                    run_id=evaluation_run_id,
                    status="failed",
                    error_message="All batch jobs failed",
                )
                return {"success": False, "error": "All batch jobs failed"}

    except Exception as e:
        logger.error(
            f"[poll_stt_batch_status] Polling failed | "
            f"run_id: {evaluation_run_id}, error: {str(e)}",
            exc_info=True,
        )

        # Re-schedule polling (might be temporary error)
        if poll_count < MAX_POLL_COUNT:
            poll_stt_batch_status.apply_async(
                kwargs={
                    "evaluation_run_id": evaluation_run_id,
                    "org_id": org_id,
                    "project_id": project_id,
                    "trace_id": trace_id,
                    "batch_jobs": batch_jobs,
                    "poll_count": poll_count + 1,
                },
                countdown=60,  # Wait longer on error
            )

        return {"success": False, "error": str(e)}


@celery_app.task(bind=True, queue="low_priority")
def process_stt_batch_results(
    self,
    evaluation_run_id: int,
    org_id: int,
    project_id: int,
    trace_id: str,
    batch_jobs: dict[str, str],
    **kwargs,
) -> dict[str, Any]:
    """Process results from completed Gemini batch jobs using GeminiBatchProvider.

    Args:
        evaluation_run_id: ID of the evaluation run
        org_id: Organization ID
        project_id: Project ID
        trace_id: Correlation/trace ID
        batch_jobs: Dict of provider -> batch_id

    Returns:
        dict: Processing result
    """
    correlation_id.set(trace_id)

    logger.info(
        f"[process_stt_batch_results] Processing batch results | "
        f"run_id: {evaluation_run_id}"
    )

    try:
        with Session(engine) as session:
            # Get the run to access sample mapping
            run = get_stt_run_by_id(
                session=session,
                run_id=evaluation_run_id,
                org_id=org_id,
                project_id=project_id,
            )

            if not run or not run.score:
                logger.error(
                    f"[process_stt_batch_results] Run or score not found | "
                    f"run_id: {evaluation_run_id}"
                )
                return {"success": False, "error": "Run data not found"}

            sample_file_mapping = run.score.get("sample_file_mapping", [])
            sample_ids = [item["sample_id"] for item in sample_file_mapping]

            # Initialize Gemini client and providers
            gemini_client = GeminiClient.from_credentials(
                session=session,
                org_id=org_id,
                project_id=project_id,
            )
            batch_provider = GeminiBatchProvider(client=gemini_client.client)
            files_manager = GeminiFilesManager(gemini_client.client)

            processed_count = 0
            failed_count = 0

            for provider, batch_id in batch_jobs.items():
                try:
                    # Use GeminiBatchProvider to download results
                    results = batch_provider.download_batch_results(batch_id)

                    logger.info(
                        f"[process_stt_batch_results] Got batch results | "
                        f"provider: {provider}, result_count: {len(results)}"
                    )

                    # Match results to samples by index
                    for batch_result in results:
                        custom_id = batch_result["custom_id"]
                        # custom_id is the index as string
                        try:
                            index = int(custom_id)
                        except (ValueError, TypeError):
                            index = results.index(batch_result)

                        if index >= len(sample_ids):
                            continue

                        sample_id = sample_ids[index]

                        # Find result record for this sample and provider
                        stmt = select(STTResult).where(
                            STTResult.evaluation_run_id == evaluation_run_id,
                            STTResult.stt_sample_id == sample_id,
                            STTResult.provider == provider,
                        )
                        result_record = session.exec(stmt).one_or_none()

                        if result_record:
                            if batch_result.get("response"):
                                text = batch_result["response"].get("text", "")
                                update_stt_result(
                                    session=session,
                                    result_id=result_record.id,
                                    transcription=text,
                                    status=STTResultStatus.COMPLETED.value,
                                )
                                processed_count += 1
                            else:
                                update_stt_result(
                                    session=session,
                                    result_id=result_record.id,
                                    status=STTResultStatus.FAILED.value,
                                    error_message=batch_result.get(
                                        "error", "Unknown error"
                                    ),
                                )
                                failed_count += 1

                except Exception as e:
                    logger.error(
                        f"[process_stt_batch_results] Failed to process provider results | "
                        f"provider: {provider}, error: {str(e)}"
                    )
                    failed_count += len(sample_file_mapping)

            # Clean up Google Files
            for item in sample_file_mapping:
                file_uri = item.get("file_uri")
                if file_uri:
                    try:
                        files_manager.delete_file(file_uri)
                    except Exception as e:
                        logger.warning(
                            f"[process_stt_batch_results] Failed to delete Google file | "
                            f"file_uri: {file_uri}, error: {str(e)}"
                        )

            # Update run status
            status_counts = count_results_by_status(
                session=session, run_id=evaluation_run_id
            )

            completed = status_counts.get(STTResultStatus.COMPLETED.value, 0)
            failed = status_counts.get(STTResultStatus.FAILED.value, 0)
            pending = status_counts.get(STTResultStatus.PENDING.value, 0)

            final_status = "completed" if pending == 0 else "processing"
            error_message = None
            if failed > 0:
                error_message = f"{failed} transcription(s) failed"

            update_stt_run(
                session=session,
                run_id=evaluation_run_id,
                status=final_status,
                processed_samples=completed + failed,
                error_message=error_message,
            )

            logger.info(
                f"[process_stt_batch_results] Batch results processed | "
                f"run_id: {evaluation_run_id}, completed: {completed}, "
                f"failed: {failed}, status: {final_status}"
            )

            return {
                "success": True,
                "run_id": evaluation_run_id,
                "completed": completed,
                "failed": failed,
                "status": final_status,
            }

    except Exception as e:
        logger.error(
            f"[process_stt_batch_results] Failed to process results | "
            f"run_id: {evaluation_run_id}, error: {str(e)}",
            exc_info=True,
        )

        with Session(engine) as session:
            update_stt_run(
                session=session,
                run_id=evaluation_run_id,
                status="failed",
                error_message=f"Result processing failed: {str(e)}",
            )

        return {"success": False, "error": str(e)}
