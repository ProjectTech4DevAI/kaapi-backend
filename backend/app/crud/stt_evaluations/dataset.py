"""CRUD operations for STT evaluation datasets and samples."""

import logging
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select, func

from app.core.exception_handlers import HTTPException
from app.core.util import now
from app.models import EvaluationDataset
from app.models.stt_evaluation import (
    EvaluationType,
    STTSample,
    STTSampleCreate,
    STTDatasetPublic,
    STTSamplePublic,
)

logger = logging.getLogger(__name__)


def create_stt_dataset(
    *,
    session: Session,
    name: str,
    org_id: int,
    project_id: int,
    description: str | None = None,
    language: str | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> EvaluationDataset:
    """Create a new STT evaluation dataset.

    Args:
        session: Database session
        name: Dataset name
        org_id: Organization ID
        project_id: Project ID
        description: Optional description
        language: Optional default language code
        dataset_metadata: Optional metadata dict

    Returns:
        EvaluationDataset: Created dataset

    Raises:
        HTTPException: If dataset with same name already exists
    """
    logger.info(
        f"[create_stt_dataset] Creating STT dataset | "
        f"name: {name}, org_id: {org_id}, project_id: {project_id}"
    )

    dataset = EvaluationDataset(
        name=name,
        description=description,
        type=EvaluationType.STT.value,
        language=language,
        dataset_metadata=dataset_metadata or {},
        organization_id=org_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )

    try:
        session.add(dataset)
        session.commit()
        session.refresh(dataset)

        logger.info(
            f"[create_stt_dataset] STT dataset created | "
            f"dataset_id: {dataset.id}, name: {name}"
        )

        return dataset

    except IntegrityError as e:
        session.rollback()
        if "uq_evaluation_dataset_name_org_project" in str(e):
            logger.error(
                f"[create_stt_dataset] Dataset name already exists | name: {name}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Dataset with name '{name}' already exists",
            )
        raise


def create_stt_samples(
    *,
    session: Session,
    dataset_id: int,
    org_id: int,
    project_id: int,
    samples: list[STTSampleCreate],
) -> list[STTSample]:
    """Create STT samples for a dataset.

    Args:
        session: Database session
        dataset_id: Parent dataset ID
        org_id: Organization ID
        project_id: Project ID
        samples: List of sample data

    Returns:
        list[STTSample]: Created samples
    """
    logger.info(
        f"[create_stt_samples] Creating STT samples | "
        f"dataset_id: {dataset_id}, sample_count: {len(samples)}"
    )

    created_samples = []

    for sample_data in samples:
        sample = STTSample(
            object_store_url=sample_data.object_store_url,
            ground_truth=sample_data.ground_truth,
            dataset_id=dataset_id,
            organization_id=org_id,
            project_id=project_id,
            inserted_at=now(),
            updated_at=now(),
        )
        session.add(sample)
        created_samples.append(sample)

    session.commit()

    # Refresh all samples to get IDs
    for sample in created_samples:
        session.refresh(sample)

    logger.info(
        f"[create_stt_samples] STT samples created | "
        f"dataset_id: {dataset_id}, created_count: {len(created_samples)}"
    )

    return created_samples


def get_stt_dataset_by_id(
    *,
    session: Session,
    dataset_id: int,
    org_id: int,
    project_id: int,
) -> EvaluationDataset | None:
    """Get an STT dataset by ID.

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
        EvaluationDataset.type == EvaluationType.STT.value,
    )

    return session.exec(statement).one_or_none()


def list_stt_datasets(
    *,
    session: Session,
    org_id: int,
    project_id: int,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[STTDatasetPublic], int]:
    """List STT datasets for a project.

    Args:
        session: Database session
        org_id: Organization ID
        project_id: Project ID
        limit: Maximum results to return
        offset: Number of results to skip

    Returns:
        tuple[list[STTDatasetPublic], int]: Datasets and total count
    """
    # Get total count
    count_stmt = select(func.count(EvaluationDataset.id)).where(
        EvaluationDataset.organization_id == org_id,
        EvaluationDataset.project_id == project_id,
        EvaluationDataset.type == EvaluationType.STT.value,
    )
    total = session.exec(count_stmt).one()

    # Get datasets
    statement = (
        select(EvaluationDataset)
        .where(
            EvaluationDataset.organization_id == org_id,
            EvaluationDataset.project_id == project_id,
            EvaluationDataset.type == EvaluationType.STT.value,
        )
        .order_by(EvaluationDataset.inserted_at.desc())
        .offset(offset)
        .limit(limit)
    )

    datasets = session.exec(statement).all()

    # Convert to public models with sample counts
    result = []
    for dataset in datasets:
        sample_count = get_sample_count_for_dataset(
            session=session, dataset_id=dataset.id
        )
        result.append(
            STTDatasetPublic(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                type=dataset.type,
                language=dataset.language,
                dataset_metadata=dataset.dataset_metadata,
                sample_count=sample_count,
                organization_id=dataset.organization_id,
                project_id=dataset.project_id,
                inserted_at=dataset.inserted_at,
                updated_at=dataset.updated_at,
            )
        )

    return result, total


def get_sample_count_for_dataset(*, session: Session, dataset_id: int) -> int:
    """Get the number of samples in a dataset.

    Args:
        session: Database session
        dataset_id: Dataset ID

    Returns:
        int: Sample count
    """
    statement = select(func.count(STTSample.id)).where(
        STTSample.dataset_id == dataset_id
    )
    return session.exec(statement).one()


def get_samples_by_dataset_id(
    *,
    session: Session,
    dataset_id: int,
    org_id: int,
    project_id: int,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[STTSample], int]:
    """Get samples for a dataset.

    Args:
        session: Database session
        dataset_id: Dataset ID
        org_id: Organization ID
        project_id: Project ID
        limit: Maximum results to return
        offset: Number of results to skip

    Returns:
        tuple[list[STTSample], int]: Samples and total count
    """
    # Get total count
    count_stmt = select(func.count(STTSample.id)).where(
        STTSample.dataset_id == dataset_id,
        STTSample.organization_id == org_id,
        STTSample.project_id == project_id,
    )
    total = session.exec(count_stmt).one()

    # Get samples
    statement = (
        select(STTSample)
        .where(
            STTSample.dataset_id == dataset_id,
            STTSample.organization_id == org_id,
            STTSample.project_id == project_id,
        )
        .order_by(STTSample.id)
        .offset(offset)
        .limit(limit)
    )

    samples = session.exec(statement).all()

    return list(samples), total


def update_dataset_metadata(
    *,
    session: Session,
    dataset_id: int,
    metadata: dict[str, Any],
) -> EvaluationDataset | None:
    """Update dataset metadata.

    Args:
        session: Database session
        dataset_id: Dataset ID
        metadata: Metadata to merge

    Returns:
        EvaluationDataset | None: Updated dataset
    """
    statement = select(EvaluationDataset).where(EvaluationDataset.id == dataset_id)
    dataset = session.exec(statement).one_or_none()

    if not dataset:
        return None

    # Merge metadata
    current_metadata = dataset.dataset_metadata or {}
    current_metadata.update(metadata)
    dataset.dataset_metadata = current_metadata
    dataset.updated_at = now()

    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    return dataset
