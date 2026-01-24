"""Generic batch operations orchestrator."""

import logging
from typing import Any

from sqlmodel import Session

from app.core.batch.base import BatchProvider
from app.core.cloud import get_cloud_storage
from app.core.storage_utils import upload_jsonl_to_object_store as shared_upload_jsonl
from app.crud.job.job import create_batch_job, update_batch_job
from app.models.batch_job import BatchJob, BatchJobCreate, BatchJobUpdate

logger = logging.getLogger(__name__)


def start_batch_job(
    session: Session,
    provider: BatchProvider,
    provider_name: str,
    job_type: str,
    organization_id: int,
    project_id: int,
    jsonl_data: list[dict[str, Any]],
    config: dict[str, Any],
) -> BatchJob:
    """
    Create and start a batch job with the specified provider.

    Creates a batch_job record, calls the provider to create the batch,
    and updates the record with provider IDs.

    Returns:
        BatchJob with provider IDs populated
    """
    logger.info(
        f"[start_batch_job] Starting | provider={provider_name} | type={job_type} | "
        f"org={organization_id} | project={project_id} | items={len(jsonl_data)}"
    )

    batch_job_create = BatchJobCreate(
        provider=provider_name,
        job_type=job_type,
        organization_id=organization_id,
        project_id=project_id,
        config=config,
        total_items=len(jsonl_data),
    )

    batch_job = create_batch_job(session=session, batch_job_create=batch_job_create)

    try:
        batch_result = provider.create_batch(jsonl_data=jsonl_data, config=config)

        batch_job_update = BatchJobUpdate(
            provider_batch_id=batch_result["provider_batch_id"],
            provider_file_id=batch_result["provider_file_id"],
            provider_status=batch_result["provider_status"],
            total_items=batch_result.get("total_items", len(jsonl_data)),
        )

        batch_job = update_batch_job(
            session=session, batch_job=batch_job, batch_job_update=batch_job_update
        )

        logger.info(
            f"[start_batch_job] Success | id={batch_job.id} | "
            f"provider_batch_id={batch_job.provider_batch_id}"
        )

        return batch_job

    except Exception as e:
        logger.error(f"[start_batch_job] Failed | {e}", exc_info=True)

        batch_job_update = BatchJobUpdate(
            error_message=f"Batch creation failed: {str(e)}"
        )
        update_batch_job(
            session=session, batch_job=batch_job, batch_job_update=batch_job_update
        )

        raise


def download_batch_results(
    provider: BatchProvider, batch_job: BatchJob
) -> list[dict[str, Any]]:
    """Download raw batch results from provider."""
    if not batch_job.provider_output_file_id:
        raise ValueError(
            f"Batch job {batch_job.id} does not have provider_output_file_id"
        )

    logger.info(
        f"[download_batch_results] Downloading | id={batch_job.id} | "
        f"output_file_id={batch_job.provider_output_file_id}"
    )

    try:
        results = provider.download_batch_results(batch_job.provider_output_file_id)

        logger.info(
            f"[download_batch_results] Downloaded | batch_job_id={batch_job.id} | "
            f"results={len(results)}"
        )

        return results

    except Exception as e:
        logger.error(f"[download_batch_results] Failed | {e}", exc_info=True)
        raise


def process_completed_batch(
    session: Session,
    provider: BatchProvider,
    batch_job: BatchJob,
    upload_to_object_store: bool = True,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Process a completed batch: download results and optionally upload to object store.

    Returns:
        Tuple of (results, object_store_url)
    """
    logger.info(f"[process_completed_batch] Processing | id={batch_job.id}")

    try:
        results = download_batch_results(provider=provider, batch_job=batch_job)

        object_store_url = None
        if upload_to_object_store:
            try:
                object_store_url = upload_batch_results_to_object_store(
                    session=session, batch_job=batch_job, results=results
                )
                logger.info(
                    f"[process_completed_batch] Uploaded to object store | {object_store_url}"
                )
            except Exception as store_error:
                logger.warning(
                    f"[process_completed_batch] Object store upload failed "
                    f"(credentials may not be configured) | {store_error}",
                    exc_info=True,
                )

        if object_store_url:
            batch_job_update = BatchJobUpdate(raw_output_url=object_store_url)
            update_batch_job(
                session=session, batch_job=batch_job, batch_job_update=batch_job_update
            )

        return results, object_store_url

    except Exception as e:
        logger.error(f"[process_completed_batch] Failed | {e}", exc_info=True)
        raise


def upload_batch_results_to_object_store(
    session: Session, batch_job: BatchJob, results: list[dict[str, Any]]
) -> str | None:
    """Upload batch results to object store."""
    logger.info(
        f"[upload_batch_results_to_object_store] Uploading | batch_job_id={batch_job.id}"
    )

    try:
        storage = get_cloud_storage(session=session, project_id=batch_job.project_id)

        subdirectory = f"{batch_job.job_type}/batch-{batch_job.id}"
        filename = "results.jsonl"

        object_store_url = shared_upload_jsonl(
            storage=storage,
            results=results,
            filename=filename,
            subdirectory=subdirectory,
        )

        return object_store_url

    except Exception as e:
        logger.error(
            f"[upload_batch_results_to_object_store] Failed | {e}", exc_info=True
        )
        raise
