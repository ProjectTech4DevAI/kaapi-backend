import pytest
from sqlmodel import Session
from fastapi import HTTPException

from app.models import FineTuningUpdate, FineTuningJobCreate, FineTuningStatus
from app.crud import (
    create_fine_tuning_job,
    fetch_by_provider_job_id,
    fetch_by_id,
    fetch_by_document_id,
    update_finetune_job,
    fetch_active_jobs_by_document_id,
)
from app.tests.utils.test_data import create_test_fine_tuning_jobs
from app.tests.utils.utils import get_project, get_document


def test_create_fine_tuning_job(db: Session) -> None:
    project = get_project(db, "Dalgo")
    document = get_document(db, "dalgo_sample.json")

    job_request = FineTuningJobCreate(
        document_id=document.id,
        base_model="gpt-4",
        split_ratio=[0.8, 0.2],
        system_prompt="you are a model able to classify",
    )

    created_job, created = create_fine_tuning_job(
        session=db,
        request=job_request,
        split_ratio=0.8,
        project_id=project.id,
        organization_id=project.organization_id,
    )

    assert created is True
    assert created_job.id is not None
    assert created_job.document_id == document.id
    assert created_job.split_ratio == 0.8
    assert created_job.project_id == project.id
    assert created_job.provider_job_id is None


def test_fetch_by_provider_job_id_success(db: Session) -> None:
    jobs, _ = create_test_fine_tuning_jobs(db, ratios=[0.3])
    job = jobs[0]

    result = fetch_by_provider_job_id(
        db, provider_job_id=job.provider_job_id, project_id=job.project_id
    )
    assert result.id == job.id


def test_fetch_by_provider_job_id_not_found(db: Session) -> None:
    with pytest.raises(HTTPException) as exc:
        fetch_by_provider_job_id(db, "invalid_id", project_id=999)
    assert exc.value.status_code == 404


def test_fetch_by_id_success(db: Session) -> None:
    jobs, _ = create_test_fine_tuning_jobs(db, ratios=[0.3, 0.4])
    job = jobs[0]

    result = fetch_by_id(db, job_id=job.id, project_id=job.project_id)
    assert result.id == job.id


def test_fetch_by_id_not_found(db: Session) -> None:
    with pytest.raises(HTTPException) as exc:
        fetch_by_id(db, job_id=9999, project_id=1)
    assert exc.value.status_code == 404


def test_fetch_by_document_id_filters(db: Session) -> None:
    jobs, _ = create_test_fine_tuning_jobs(db, ratios=[0.3, 0.4])
    job = jobs[0]

    results = fetch_by_document_id(
        session=db,
        document_id=job.document_id,
        project_id=job.project_id,
        split_ratio=job.split_ratio,
        base_model=job.base_model,
    )
    assert len(results) == 1
    assert results[0].id == job.id


def test_update_finetune_job(db: Session) -> None:
    jobs, _ = create_test_fine_tuning_jobs(db, ratios=[0.3, 0.4])
    job = jobs[0]

    update = FineTuningUpdate(status="completed", fine_tuned_model="ft:gpt-4:custom")
    updated_job = update_finetune_job(db, job=job, update=update)

    assert updated_job.status == "completed"
    assert updated_job.fine_tuned_model == "ft:gpt-4:custom"


def test_fetch_active_jobs_by_document_id(db: Session) -> None:
    project = get_project(db, "Dalgo")
    document = get_document(db, "dalgo_sample.json")

    job_request = FineTuningJobCreate(
        document_id=document.id,
        base_model="gpt-4",
        split_ratio=[0.3, 0.5, 0.7, 0.9],
        system_prompt="you are a model good at classification",
    )
    active_job, _ = create_fine_tuning_job(
        db,
        request=job_request,
        split_ratio=0.3,
        project_id=project.id,
        organization_id=project.organization_id,
        status=FineTuningStatus.running,
    )

    failed_job, _ = create_fine_tuning_job(
        db,
        request=job_request,
        split_ratio=0.5,
        project_id=project.id,
        organization_id=project.organization_id,
        status=FineTuningStatus.failed,
    )

    result = fetch_active_jobs_by_document_id(
        session=db,
        document_id=document.id,
        project_id=project.id,
    )

    assert len(result) == 1
    assert result[0].id == active_job.id
    assert result[0].status == FineTuningStatus.running
    assert result[0].is_deleted is False
