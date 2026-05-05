import logging
from sqlmodel import Session
from uuid import UUID

from app.models.job import Job, JobType, JobUpdate
from app.core.util import now

logger = logging.getLogger(__name__)


class JobCrud:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        job_type: JobType,
        trace_id: str | None = None,
        project_id: int | None = None,
    ) -> Job:
        new_job = Job(job_type=job_type, trace_id=trace_id, project_id=project_id)
        self.session.add(new_job)
        self.session.commit()
        self.session.refresh(new_job)
        return new_job

    def update(self, job_id: UUID, job_update: JobUpdate) -> Job:
        job = self.session.get(Job, job_id)
        if not job:
            raise ValueError(f"Job not found with the given job_id {job_id}")

        update_data = job_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(job, field, value)

        job.updated_at = now()
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)

        return job

    def get(self, job_id: UUID, project_id: int) -> Job | None:
        job = self.session.get(Job, job_id)
        if job is None:
            return None
        if job.project_id not in (None, project_id):
            return None
        return job
