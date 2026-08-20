import logging
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session, and_, col, select

from app.core.util import now
from app.models import (
    DocTransformationJob,
    DocTransformJobCreate,
    DocTransformJobUpdate,
)
from app.models.document import Document

logger = logging.getLogger(__name__)


class DocTransformationJobCrud:
    def __init__(self, session: Session, project_id: int):
        self.session = session
        self.project_id = project_id

    def create(self, payload: DocTransformJobCreate) -> DocTransformationJob:
        job = DocTransformationJob(**payload.model_dump())
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        logger.info(
            f"[DocTransformationJobCrud.create] Created new transformation job | id: {job.id}, source_document_id: {job.source_document_id}"
        )
        return job

    def read_one(self, job_id: UUID) -> DocTransformationJob:
        statement = (
            select(DocTransformationJob)
            .join(
                Document,
                col(DocTransformationJob.source_document_id) == col(Document.id),
            )
            .where(
                and_(
                    DocTransformationJob.id == job_id,
                    Document.project_id == self.project_id,
                    col(Document.deleted_at).is_(None),
                )
            )
        )

        job = self.session.exec(statement).one_or_none()
        if not job:
            logger.warning(
                f"[DocTransformationJobCrud.read_one] Job not found or Document is deleted | id: {job_id}, project_id: {self.project_id}"
            )
            raise HTTPException(status_code=404, detail="Transformation job not found")
        return job

    def read_each(self, job_ids: set[UUID]) -> list[DocTransformationJob]:
        statement = (
            select(DocTransformationJob)
            .join(
                Document,
                col(DocTransformationJob.source_document_id) == col(Document.id),
            )
            .where(
                and_(
                    col(DocTransformationJob.id).in_(list(job_ids)),
                    Document.project_id == self.project_id,
                    col(Document.deleted_at).is_(None),
                )
            )
        )

        jobs = self.session.exec(statement).all()
        return list(jobs)

    def update(
        self,
        job_id: UUID,
        patch: DocTransformJobUpdate,
    ) -> DocTransformationJob:
        """Update an existing doc transformation job and return the updated row."""
        job = self.read_one(job_id)

        # Only apply fields that were explicitly set and not None
        changes = patch.model_dump(exclude_unset=True, exclude_none=True)
        for field, value in changes.items():
            setattr(job, field, value)

        job.updated_at = now()

        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)

        logger.info(
            f"[DocTransformationJobCrud.update_status] Updated job status | id: {job.id}"
        )
        return job
