"""Batch status polling operations."""

import logging
from typing import Any

from sqlmodel import Session

from app.core.batch.base import BatchProvider
from app.crud.job.job import update_batch_job
from app.models.batch_job import BatchJob, BatchJobUpdate

logger = logging.getLogger(__name__)


def poll_batch_status(
    session: Session, provider: BatchProvider, batch_job: BatchJob
) -> dict[str, Any]:
    """Poll provider for batch status and update database."""
    logger.info(
        f"[poll_batch_status] Polling | id={batch_job.id} | "
        f"provider_batch_id={batch_job.provider_batch_id}"
    )

    try:
        status_result = provider.get_batch_status(batch_job.provider_batch_id)

        provider_status = status_result["provider_status"]
        if provider_status != batch_job.provider_status:
            update_data = {"provider_status": provider_status}

            if status_result.get("provider_output_file_id"):
                update_data["provider_output_file_id"] = status_result[
                    "provider_output_file_id"
                ]

            if status_result.get("error_message"):
                update_data["error_message"] = status_result["error_message"]

            batch_job_update = BatchJobUpdate(**update_data)
            batch_job = update_batch_job(
                session=session, batch_job=batch_job, batch_job_update=batch_job_update
            )

            logger.info(
                f"[poll_batch_status] Updated | id={batch_job.id} | "
                f"{batch_job.provider_status} -> {provider_status}"
            )

        return status_result

    except Exception as e:
        logger.error(f"[poll_batch_status] Failed | {e}", exc_info=True)
        raise
