"""CRUD operations for TTS evaluation datasets."""

import logging
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, func, select

from app.core.exception_handlers import HTTPException
from app.core.util import now
from app.models import EvaluationDataset
from app.models.stt_evaluation import EvaluationType
from app.models.tts_evaluation import TTSDatasetPublic

logger = logging.getLogger(__name__)


def create_tts_dataset(
    *,
    session: Session,
    name: str,
    org_id: int,
    project_id: int,
    description: str | None = None,
    language_id: int | None = None,
    object_store_url: str | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> EvaluationDataset:
    """Create a new TTS evaluation dataset.

    Args:
        session: Database session
        name: Dataset name
        org_id: Organization ID
        project_id: Project ID
        description: Optional description
        language_id: Optional reference to global.languages table
        object_store_url: Optional object store URL
        dataset_metadata: Optional metadata dict

    Returns:
        EvaluationDataset: Created dataset

    Raises:
        HTTPException: If dataset with same name already exists
    """
    logger.info(
        f"[create_tts_dataset] Creating TTS dataset | "
        f"name: {name}, org_id: {org_id}, project_id: {project_id}"
    )

    dataset = EvaluationDataset(
        name=name,
        description=description,
        type=EvaluationType.TTS.value,
        language_id=language_id,
        object_store_url=object_store_url,
        dataset_metadata=dataset_metadata or {},
        organization_id=org_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )

    try:
        session.add(dataset)
        session.flush()

        logger.info(
            f"[create_tts_dataset] TTS dataset created | "
            f"dataset_id: {dataset.id}, name: {name}"
        )

        return dataset

    except IntegrityError as e:
        session.rollback()
        if "uq_evaluation_dataset_name_org_project" in str(e):
            logger.error(
                f"[create_tts_dataset] Dataset name already exists | name: {name}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Dataset with name '{name}' already exists",
            )
        raise


def get_tts_dataset_by_id(
    *,
    session: Session,
    dataset_id: int,
    org_id: int,
    project_id: int,
) -> EvaluationDataset | None:
    """Get a TTS dataset by ID.

    Args:
        session: Database session
        dataset_id: Dataset ID
        org_id: Organization ID
        project_id: Project ID

    Returns:
        EvaluationDataset | None: Dataset if found
    """
    statement = select(EvaluationDataset).where(
        EvaluationDataset.id == dataset_id,
        EvaluationDataset.organization_id == org_id,
        EvaluationDataset.project_id == project_id,
        EvaluationDataset.type == EvaluationType.TTS.value,
    )

    return session.exec(statement).one_or_none()


def list_tts_datasets(
    *,
    session: Session,
    org_id: int,
    project_id: int,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[TTSDatasetPublic], int]:
    """List TTS datasets for a project.

    Args:
        session: Database session
        org_id: Organization ID
        project_id: Project ID
        limit: Maximum results to return
        offset: Number of results to skip

    Returns:
        tuple[list[TTSDatasetPublic], int]: Datasets and total count
    """
    base_filter = (
        EvaluationDataset.organization_id == org_id,
        EvaluationDataset.project_id == project_id,
        EvaluationDataset.type == EvaluationType.TTS.value,
    )

    count_stmt = select(func.count(EvaluationDataset.id)).where(*base_filter)
    total = session.exec(count_stmt).one()

    statement = (
        select(EvaluationDataset)
        .where(*base_filter)
        .order_by(EvaluationDataset.inserted_at.desc())
        .offset(offset)
        .limit(limit)
    )

    datasets = session.exec(statement).all()

    result = [
        TTSDatasetPublic(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            type=dataset.type,
            language_id=dataset.language_id,
            object_store_url=dataset.object_store_url,
            dataset_metadata=dataset.dataset_metadata,
            organization_id=dataset.organization_id,
            project_id=dataset.project_id,
            inserted_at=dataset.inserted_at,
            updated_at=dataset.updated_at,
        )
        for dataset in datasets
    ]

    return result, total
