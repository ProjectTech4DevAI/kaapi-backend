"""CRUD operations for assessment datasets."""

import logging
from typing import Any

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.core.util import now
from app.models.assessment import Assessment
from app.models.evaluation import EvaluationDataset
from app.models.stt_evaluation import EvaluationType

logger = logging.getLogger(__name__)


def create_assessment_dataset(
    *,
    session: Session,
    name: str,
    dataset_metadata: dict[str, Any],
    organization_id: int,
    project_id: int,
    description: str | None = None,
    object_store_url: str | None = None,
) -> EvaluationDataset:
    """Create an assessment dataset backed by the shared evaluation_dataset table."""
    dataset = EvaluationDataset(
        name=name,
        description=description,
        type=EvaluationType.ASSESSMENT.value,
        dataset_metadata=dataset_metadata,
        object_store_url=object_store_url,
        langfuse_dataset_id=None,
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )

    try:
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
    except IntegrityError as e:
        session.rollback()
        logger.error(
            "[create_assessment_dataset] Dataset name already exists | "
            "name=%s | org_id=%s | project_id=%s",
            name,
            organization_id,
            project_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"Dataset with name '{name}' already exists in this "
                "organization and project. Please choose a different name."
            ),
        ) from e
    except Exception as e:
        session.rollback()
        logger.error(
            "[create_assessment_dataset] Failed to create dataset | name=%s",
            name,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save assessment dataset metadata: {e}",
        ) from e

    logger.info(
        "[create_assessment_dataset] Created assessment dataset | "
        "id=%s | name=%s | org_id=%s | project_id=%s",
        dataset.id,
        name,
        organization_id,
        project_id,
    )
    return dataset


def get_assessment_dataset_by_id(
    *,
    session: Session,
    dataset_id: int,
    organization_id: int,
    project_id: int,
) -> EvaluationDataset | None:
    """Fetch an assessment dataset by ID, scoped to organization and project."""
    statement = (
        select(EvaluationDataset)
        .where(EvaluationDataset.id == dataset_id)
        .where(EvaluationDataset.organization_id == organization_id)
        .where(EvaluationDataset.project_id == project_id)
        .where(EvaluationDataset.type == EvaluationType.ASSESSMENT.value)
    )
    return session.exec(statement).first()


def list_assessment_datasets(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[EvaluationDataset]:
    """List assessment datasets for an organization and project."""
    statement = (
        select(EvaluationDataset)
        .where(EvaluationDataset.organization_id == organization_id)
        .where(EvaluationDataset.project_id == project_id)
        .where(EvaluationDataset.type == EvaluationType.ASSESSMENT.value)
        .order_by(EvaluationDataset.inserted_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.exec(statement).all())


def delete_assessment_dataset(
    *, session: Session, dataset: EvaluationDataset
) -> str | None:
    """Delete an unused assessment dataset."""
    statement = select(Assessment).where(Assessment.dataset_id == dataset.id)
    assessments = session.exec(statement).all()
    if assessments:
        return (
            f"Cannot delete dataset {dataset.id}: it is being used by "
            f"{len(assessments)} assessment(s). Please delete the assessments first."
        )

    try:
        dataset_id = dataset.id
        dataset_name = dataset.name
        session.delete(dataset)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(
            "[delete_assessment_dataset] Failed to delete dataset | dataset_id=%s",
            dataset.id,
            exc_info=True,
        )
        return f"Failed to delete dataset: {e}"

    logger.info(
        "[delete_assessment_dataset] Deleted assessment dataset | id=%s | name=%s",
        dataset_id,
        dataset_name,
    )
    return None
