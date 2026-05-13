from uuid import UUID
import logging
from typing import Optional

from sqlmodel import Session, select
from fastapi import HTTPException

from app.core.util import now
from app.models import (
    Fine_Tuning,
    FineTuningJobCreate,
    FineTuningUpdate,
    FineTuningStatus,
)
from app.crud import DocumentCrud

logger = logging.getLogger(__name__)


def create_fine_tuning_job(
    session: Session,
    request: FineTuningJobCreate,
    split_ratio: float,
    status: FineTuningStatus = FineTuningStatus.pending,
    project_id: int = None,
    organization_id: int = None,
) -> tuple[Fine_Tuning, bool]:
    active_jobs = fetch_active_jobs_by_document_id(
        session=session,
        document_id=request.document_id,
        project_id=project_id,
        split_ratio=split_ratio,
        base_model=request.base_model,
    )
    if active_jobs:
        existing = active_jobs[0]
        logger.info(
            f"[Create_fine_tuning_job]Active fine-tune job exists; returning it. document_id={request.document_id}, split_ratio={split_ratio}, base_model={request.base_model}, project_id={project_id}"
        )
        return existing, False

    document_crud = DocumentCrud(
        session, project_id
    )  # to check if the given document is present in the document table or not
    document = document_crud.read_one(request.document_id)

    fine_tune_data = request.model_dump(exclude_unset=True)
    base_data = {
        **fine_tune_data,
        "split_ratio": split_ratio,
        "project_id": project_id,
        "organization_id": organization_id,
        "status": status,
    }

    fine_tune = Fine_Tuning(**base_data)
    fine_tune.updated_at = now()

    session.add(fine_tune)
    session.commit()
    session.refresh(fine_tune)

    logger.info(
        f"[Create_fine_tuning_job]Created new fine-tuning job ID={fine_tune.id}, project_id={project_id}"
    )
    return fine_tune, True


def fetch_by_provider_job_id(
    session: Session, provider_job_id: str, project_id: int
) -> Fine_Tuning:
    job = session.exec(
        select(Fine_Tuning).where(
            Fine_Tuning.provider_job_id == provider_job_id,
            Fine_Tuning.project_id == project_id,
        )
    ).one_or_none()

    if job is None:
        logger.warning(
            f"Fine-tune job not found for openai_job_id={provider_job_id}, project_id={project_id}"
        )
        raise HTTPException(status_code=404, detail="OpenAI fine tuning ID not found")

    return job


def fetch_by_id(session: Session, job_id: int, project_id: int) -> Fine_Tuning:
    job = session.exec(
        select(Fine_Tuning).where(
            Fine_Tuning.id == job_id, Fine_Tuning.project_id == project_id
        )
    ).one_or_none()

    if job is None:
        logger.warning(
            f"[fetch_by_id]Fine-tune job not found: job_id={job_id}, project_id={project_id}"
        )
        raise HTTPException(status_code=404, detail="Job not found")

    logger.info(
        f"[fetch_by_id]Fetched fine-tune job ID={job.id}, project_id={project_id}"
    )
    return job


def fetch_by_document_id(
    session: Session,
    document_id: UUID,
    project_id: int,
    split_ratio: float = None,
    base_model: Optional[str] = None,
) -> list[Fine_Tuning]:
    query = select(Fine_Tuning).where(
        Fine_Tuning.document_id == document_id, Fine_Tuning.project_id == project_id
    )

    if split_ratio is not None:
        query = query.where(Fine_Tuning.split_ratio == split_ratio)
    if base_model is not None:
        query = query.where(Fine_Tuning.base_model == base_model)

    jobs = session.exec(query).all()
    logger.info(
        f"[fetch_by_document_id]Found {len(jobs)} fine-tune jobs for document_id={document_id}, project_id={project_id}"
    )
    return jobs


def fetch_active_jobs_by_document_id(
    session: Session,
    document_id: UUID,
    project_id: int,
    split_ratio: Optional[float] = None,
    base_model: Optional[str] = None,
    exclude_job_id: Optional[int] = None,
) -> list["Fine_Tuning"]:
    """
    Return all ACTIVE jobs for the given document & project.
    Active = status != failed AND is_deleted is false.
    """
    stmt = (
        select(Fine_Tuning)
        .where(
            Fine_Tuning.document_id == document_id,
            Fine_Tuning.project_id == project_id,
            Fine_Tuning.is_deleted.is_(False),
            Fine_Tuning.status != FineTuningStatus.failed,
        )
        .order_by(Fine_Tuning.inserted_at.desc())
    )

    if split_ratio is not None:
        stmt = stmt.where(Fine_Tuning.split_ratio == split_ratio)

    if base_model is not None:
        stmt = stmt.where(Fine_Tuning.base_model == base_model)

    if exclude_job_id is not None:
        stmt = stmt.where(Fine_Tuning.id != exclude_job_id)

    return session.exec(stmt).all()


def update_finetune_job(
    session: Session,
    job: Fine_Tuning,
    update: FineTuningUpdate,
) -> Fine_Tuning:
    for key, value in update.model_dump(exclude_unset=True).items():
        setattr(job, key, value)

    job.updated_at = now()

    session.add(job)
    session.commit()
    session.refresh(job)

    logger.info(
        f"[update_finetune_job]Updated fine-tune job ID={job.id}, project_id={job.project_id}"
    )
    return job
