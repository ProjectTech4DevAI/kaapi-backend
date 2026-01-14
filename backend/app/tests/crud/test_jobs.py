from uuid import uuid4

import pytest
from sqlmodel import Session

from app.crud import JobCrud
from app.models import JobUpdate, JobStatus, JobType


@pytest.fixture
def dummy_jobs(db: Session):
    """Create and return a list of dummy jobs for testing."""
    crud = JobCrud(db)

    jobs = [
        crud.create(job_type=JobType.RESPONSE, trace_id="trace-1"),
        crud.create(job_type=JobType.RESPONSE, trace_id="trace-2"),
        crud.create(job_type=JobType.RESPONSE, trace_id="trace-3"),
    ]

    return jobs


def test_create_job(db: Session) -> None:
    crud = JobCrud(db)
    job = crud.create(job_type=JobType.RESPONSE, trace_id="trace-123")

    assert job.id is not None
    assert job.trace_id == "trace-123"
    assert job.status == JobStatus.PENDING


def test_get_job(db: Session, dummy_jobs) -> None:
    crud = JobCrud(db)
    job = dummy_jobs[0]

    fetched = crud.get(job.id)
    assert fetched is not None
    assert fetched.id == job.id
    assert fetched.trace_id == "trace-1"


def test_update_job(db: Session, dummy_jobs) -> None:
    crud = JobCrud(db)
    job = dummy_jobs[1]

    update_data = JobUpdate(status=JobStatus.FAILED, error_message="Error occurred")
    updated_job = crud.update(job.id, update_data)

    assert updated_job.status == JobStatus.FAILED
    assert updated_job.error_message == "Error occurred"
    assert updated_job.updated_at is not None
    assert updated_job.updated_at >= job.updated_at


def test_update_job_not_found(db: Session) -> None:
    crud = JobCrud(db)
    fake_id = uuid4()
    update_data = JobUpdate(status=JobStatus.SUCCESS)

    with pytest.raises(ValueError, match=str(fake_id)):
        crud.update(fake_id, update_data)
