from typing import Any
from uuid import uuid4

import pytest
from sqlmodel import Session
from sqlalchemy.exc import IntegrityError

from app.models import CollectionJob, CollectionJobStatus, CollectionActionType, Project
from app.crud import CollectionJobCrud
from app.core.util import now
from app.tests.utils.utils import get_project


def create_sample_collection_job(
    db: Session,
    project_id: int,
    action_type: CollectionActionType = CollectionActionType.CREATE,
    status: CollectionJobStatus = CollectionJobStatus.PENDING,
) -> CollectionJob:
    collection_job = CollectionJob(
        id=uuid4(),
        project_id=project_id,
        action_type=action_type,
        status=status,
        inserted_at=now(),
        updated_at=now(),
    )

    collection_job_crud = CollectionJobCrud(db, project_id)
    created_job = collection_job_crud.create(collection_job)

    return created_job


@pytest.fixture
def sample_project(db: Session) -> Project:
    """Fixture to create a sample project."""
    return get_project(db)


def test_create_collection_job(db: Session, sample_project: Project) -> None:
    """Test case to create a CollectionJob."""
    collection_job = CollectionJob(
        id=uuid4(),
        project_id=sample_project.id,
        action_type=CollectionActionType.CREATE,
        status=CollectionJobStatus.PENDING,
        inserted_at=now(),
        updated_at=now(),
    )
    collection_job_crud = CollectionJobCrud(db, sample_project.id)

    created_job = collection_job_crud.create(collection_job)

    assert created_job.id is not None
    assert created_job.project_id == sample_project.id
    assert created_job.action_type == CollectionActionType.CREATE
    assert created_job.status == CollectionJobStatus.PENDING
    assert created_job.inserted_at is not None
    assert created_job.updated_at is not None


def test_read_one_collection_job(db: Session, sample_project: Project) -> None:
    """Test case to read a single CollectionJob by ID."""
    collection_job = create_sample_collection_job(db, sample_project.id)

    collection_job_crud = CollectionJobCrud(db, sample_project.id)

    retrieved_job = collection_job_crud.read_one(str(collection_job.id))

    assert retrieved_job.id == collection_job.id
    assert retrieved_job.project_id == sample_project.id
    assert retrieved_job.action_type == collection_job.action_type
    assert retrieved_job.status == collection_job.status
    assert retrieved_job.inserted_at == collection_job.inserted_at


def test_read_all_collection_jobs(db: Session, sample_project: Project) -> None:
    """Test case to retrieve all collection jobs for a project."""
    collection_job1 = create_sample_collection_job(db, sample_project.id)
    collection_job2 = create_sample_collection_job(db, sample_project.id)

    db.commit()

    collection_job_crud = CollectionJobCrud(db, sample_project.id)

    collection_jobs = collection_job_crud.read_all()

    assert len(collection_jobs) == 2
    job_ids = [str(job.id) for job in collection_jobs]
    assert str(collection_job1.id) in job_ids
    assert str(collection_job2.id) in job_ids


def test_update_collection_job(db: Session, sample_project: Project) -> None:
    """Test case to update a CollectionJob."""
    collection_job = create_sample_collection_job(db, sample_project.id)

    collection_job_crud = CollectionJobCrud(db, sample_project.id)

    collection_job.status = CollectionJobStatus.FAILED
    collection_job.error_message = "model name not valid"
    collection_job.updated_at = now()

    updated_job = collection_job_crud.update(collection_job.id, collection_job)

    assert updated_job.status == CollectionJobStatus.FAILED
    assert updated_job.error_message is not None
    assert updated_job.updated_at is not None


def test_create_collection_job_with_invalid_data(
    db: Session, sample_project: Project
) -> None:
    """Test case to handle invalid data during job creation."""
    collection_job = CollectionJob(
        id=uuid4(),
        project_id=sample_project.id,
        action_type=None,
        status=CollectionJobStatus.PENDING,
        inserted_at=now(),
        updated_at=now(),
    )

    collection_job_crud = CollectionJobCrud(db, sample_project.id)

    with pytest.raises(IntegrityError):
        collection_job_crud.create(collection_job)
