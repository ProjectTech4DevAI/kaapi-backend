"""Batch status polling operations."""

import logging
from typing import Any

from sqlmodel import Session

from app.core.batch.base import BatchProvider
from app.crud.job.job import update_batch_job
from app.models.batch_job import BatchJob, BatchJobUpdate

logger = logging.getLogger(__name__)

# Status-result key -> batch_job column; the provider's transient `error_file_id` is ours.
_STATUS_RESULT_TO_COLUMN = {
    "provider_status": "provider_status",
    "provider_output_file_id": "provider_output_file_id",
    "error_file_id": "provider_error_file_id",
    "error_message": "error_message",
}


def _changed_columns(
    status_result: dict[str, Any], batch_job: BatchJob
) -> dict[str, str]:
    """Columns whose polled value differs from the row's.

    A field the provider omits (or reports empty) is left alone rather than nulled —
    OpenAI only fills the file ids once the batch reaches a terminal state.
    """
    changed: dict[str, str] = {}
    for result_key, column in _STATUS_RESULT_TO_COLUMN.items():
        value = status_result.get(result_key)
        if value and value != getattr(batch_job, column):
            changed[column] = value
    return changed


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

        # Per field, not gated on a status flip: an error file landing mid-status was dropped.
        changed = _changed_columns(status_result, batch_job)
        if changed:
            previous_status = batch_job.provider_status
            batch_job = update_batch_job(
                session=session,
                batch_job=batch_job,
                batch_job_update=BatchJobUpdate.model_validate(changed),
            )

            logger.info(
                f"[poll_batch_status] Updated | id={batch_job.id} | "
                f"fields={sorted(changed)} | "
                f"{previous_status} -> {batch_job.provider_status}"
            )

        return status_result

    except Exception as e:
        logger.error(f"[poll_batch_status] Failed | {e}", exc_info=True)
        raise
