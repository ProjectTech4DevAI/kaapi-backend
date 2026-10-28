from uuid import UUID
import logging
from typing import List

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, select

from app.models.collection_job import (
    CollectionActionType,
    CollectionJob,
    CollectionJobStatus,
    CollectionJobUpdate,
    CollectionJobCreate,
)
from app.core.util import now


logger = logging.getLogger(__name__)


class CollectionJobCrud:
    def __init__(self, session: Session, project_id: int):
        self.session = session
        self.project_id = project_id

    def read_one(self, job_id: UUID) -> CollectionJob:
        """Retrieve a single collection job by its id; 404 if not found."""
        statement = select(CollectionJob).where(
            CollectionJob.project_id == self.project_id,
            CollectionJob.id == job_id,
        )
        collection_job = self.session.exec(statement).one_or_none()
        if collection_job is None:
            logger.warning(
                "[CollectionJobCrud.read_one] Collection job not found | "
                f"{{'project_id': '{self.project_id}', 'job_id': '{job_id}'}}"
            )
            raise HTTPException(
                status_code=404,
                detail="Collection job not found",
            )

        logger.info(
            "[CollectionJobCrud.read_one] Retrieved collection job | "
            f"{{'job_id': '{job_id}'}}"
        )
        return collection_job

    def read_all(self) -> List[CollectionJob]:
        """Retrieve all collection jobs for a given project."""
        statement = select(CollectionJob).where(
            CollectionJob.project_id == self.project_id
        )
        collection_jobs = self.session.exec(statement).all()
        logger.info(
            f"[CollectionJobCrud.read_all] Retrieved all collection jobs for project | {{'project_id': '{self.project_id}', 'count': {len(collection_jobs)}}}"
        )
        return list(collection_jobs)

    def update(self, job_id: UUID, patch: CollectionJobUpdate) -> CollectionJob:
        """Update an existing collection job and return the updated row."""
        job = self.read_one(job_id)

        changes = patch.model_dump(exclude_unset=True, exclude_none=True)
        for field, value in changes.items():
            setattr(job, field, value)

        job.updated_at = now()

        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)

        logger.info(
            "[CollectionJobCrud.update] Collection job updated successfully | {'collection_job_id': '%s'}",
            job.id,
        )
        return job

    def create(self, collection_job: CollectionJobCreate) -> CollectionJob:
        """Create a new collection job."""
        try:
            job = CollectionJob(**collection_job.model_dump())
            self.session.add(job)
            self.session.commit()
            self.session.refresh(job)
            logger.info(
                f"[CollectionJobCrud.create] Collection job created successfully | {{'collection_job_id': '{job.id}'}}"
            )

        except Exception as e:
            logger.error(
                f"[CollectionJobCrud.create] Error during job creation: {str(e)}",
                exc_info=True,
            )
            raise

        return job

    def read_in_flight_create(self, name: str) -> CollectionJob | None:
        """Return the PENDING/PROCESSING CREATE job holding this name, if any."""
        statement = select(CollectionJob).where(
            CollectionJob.project_id == self.project_id,
            CollectionJob.name == name,
            CollectionJob.action_type == CollectionActionType.CREATE,
            col(CollectionJob.status).in_(
                [CollectionJobStatus.PENDING, CollectionJobStatus.PROCESSING]
            ),
        )
        return self.session.exec(statement).first()

    def create_or_get_in_flight(
        self, collection_job: CollectionJobCreate
    ) -> tuple[CollectionJob, bool]:
        """Create a CREATE job, or return the in-flight job with the same name.

        The bool is True only when a new row was inserted, so callers enqueue work once.
        """
        if collection_job.name:
            existing = self.read_in_flight_create(collection_job.name)
            if existing:
                return existing, False

        try:
            job = CollectionJob(**collection_job.model_dump())
            self.session.add(job)
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            existing = (
                self.read_in_flight_create(collection_job.name)
                if collection_job.name
                else None
            )
            if existing is None:
                raise
            return existing, False

        self.session.refresh(job)
        return job, True
