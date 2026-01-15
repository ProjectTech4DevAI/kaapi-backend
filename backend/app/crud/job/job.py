"""CRUD operations for batch_job table."""

import logging

from sqlmodel import Session, select

from app.core.util import now
from app.models.batch_job import BatchJob, BatchJobCreate, BatchJobUpdate

logger = logging.getLogger(__name__)


def create_batch_job(
    session: Session,
    batch_job_create: BatchJobCreate,
) -> BatchJob:
    """
    Create a new batch job record.

    Args:
        session: Database session
        batch_job_create: BatchJobCreate schema with all required fields

    Returns:
        Created BatchJob object

    Raises:
        Exception: If creation fails
    """
    logger.info(
        f"[create_batch_job] Creating batch job | "
        f"provider={batch_job_create.provider} | "
        f"job_type={batch_job_create.job_type} | "
        f"org_id={batch_job_create.organization_id} | "
        f"project_id={batch_job_create.project_id}"
    )

    try:
        batch_job = BatchJob.model_validate(batch_job_create)
        batch_job.inserted_at = now()
        batch_job.updated_at = now()

        session.add(batch_job)
        session.commit()
        session.refresh(batch_job)

        logger.info(f"[create_batch_job] Created batch job | id={batch_job.id}")

        return batch_job

    except Exception as e:
        logger.error(
            f"[create_batch_job] Failed to create batch job | {e}", exc_info=True
        )
        session.rollback()
        raise


def get_batch_job(session: Session, batch_job_id: int) -> BatchJob | None:
    """
    Get a batch job by ID.

    Args:
        session: Database session
        batch_job_id: Batch job ID

    Returns:
        BatchJob object if found, None otherwise
    """
    statement = select(BatchJob).where(BatchJob.id == batch_job_id)
    batch_job = session.exec(statement).first()

    return batch_job


def update_batch_job(
    session: Session,
    batch_job: BatchJob,
    batch_job_update: BatchJobUpdate,
) -> BatchJob:
    """
    Update a batch job record.

    Args:
        session: Database session
        batch_job: BatchJob object to update
        batch_job_update: BatchJobUpdate schema with fields to update

    Returns:
        Updated BatchJob object

    Raises:
        Exception: If update fails
    """
    logger.info(f"[update_batch_job] Updating batch job | id={batch_job.id}")

    try:
        # Update fields if provided
        update_data = batch_job_update.model_dump(exclude_unset=True)

        for key, value in update_data.items():
            setattr(batch_job, key, value)

        batch_job.updated_at = now()

        session.add(batch_job)
        session.commit()
        session.refresh(batch_job)

        logger.info(f"[update_batch_job] Updated batch job | id={batch_job.id}")

        return batch_job

    except Exception as e:
        logger.error(
            f"[update_batch_job] Failed to update batch job | id={batch_job.id} | {e}",
            exc_info=True,
        )
        session.rollback()
        raise


def get_batch_jobs_by_ids(
    session: Session,
    batch_job_ids: list[int],
) -> list[BatchJob]:
    """
    Get batch jobs by their IDs.

    This is used by parent tables to get their associated batch jobs for polling.

    Args:
        session: Database session
        batch_job_ids: List of batch job IDs

    Returns:
        List of BatchJob objects
    """
    if not batch_job_ids:
        return []

    statement = select(BatchJob).where(BatchJob.id.in_(batch_job_ids))
    results = session.exec(statement).all()

    logger.info(
        f"[get_batch_jobs_by_ids] Found batch jobs | found={len(results)} | requested={len(batch_job_ids)}"
    )

    return list(results)


def get_batches_by_type(
    session: Session,
    job_type: str,
    organization_id: int | None = None,
    project_id: int | None = None,
    provider_status: str | None = None,
) -> list[BatchJob]:
    """
    Get batch jobs by type with optional filters.

    Args:
        session: Database session
        job_type: Job type (e.g., "evaluation", "classification")
        organization_id: Optional filter by organization ID
        project_id: Optional filter by project ID
        provider_status: Optional filter by provider status

    Returns:
        List of BatchJob objects matching filters
    """
    statement = select(BatchJob).where(BatchJob.job_type == job_type)

    if organization_id:
        statement = statement.where(BatchJob.organization_id == organization_id)

    if project_id:
        statement = statement.where(BatchJob.project_id == project_id)

    if provider_status:
        statement = statement.where(BatchJob.provider_status == provider_status)

    results = session.exec(statement).all()

    logger.info(
        f"[get_batches_by_type] Found batch jobs | "
        f"count={len(results)} | "
        f"job_type={job_type} | "
        f"org_id={organization_id} | "
        f"project_id={project_id} | "
        f"provider_status={provider_status}"
    )

    return list(results)


def delete_batch_job(session: Session, batch_job: BatchJob) -> None:
    """
    Delete a batch job record.

    Args:
        session: Database session
        batch_job: BatchJob object to delete

    Raises:
        Exception: If deletion fails
    """
    logger.info(f"[delete_batch_job] Deleting batch job | id={batch_job.id}")

    try:
        session.delete(batch_job)
        session.commit()

        logger.info(f"[delete_batch_job] Deleted batch job | id={batch_job.id}")

    except Exception as e:
        logger.error(
            f"[delete_batch_job] Failed to delete batch job | id={batch_job.id} | {e}",
            exc_info=True,
        )
        session.rollback()
        raise
