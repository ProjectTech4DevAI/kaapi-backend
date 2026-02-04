"""CRUD operations for STT evaluation datasets and samples."""

import logging
from typing import Any
from urllib.parse import urlparse
import os

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
    object_store_url: str | None = None,
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
        object_store_url: Optional object store URL
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
        object_store_url=object_store_url,
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


def _extract_metadata_from_url(url: str) -> dict[str, Any]:
    """Extract filename and extension from S3 URL.

    Args:
        url: S3 URL of the audio file

    Returns:
        dict with original_filename and file_extension
    """
    try:
        parsed = urlparse(url)
        path = parsed.path
        filename = os.path.basename(path)
        _, extension = os.path.splitext(filename)
        # Remove leading dot from extension
        extension = extension.lstrip(".").lower() if extension else None

        return {
            "original_filename": filename if filename else None,
            "file_extension": extension,
        }
    except Exception as e:
        logger.warning(f"[_extract_metadata_from_url] Failed to extract metadata: {e}")
        return {}


def create_stt_samples(
    *,
    session: Session,
    dataset: EvaluationDataset,
    samples: list[STTSampleCreate],
) -> list[STTSample]:
    """Create STT samples for a dataset.

    Args:
        session: Database session
        dataset: Parent dataset (must have sample_count in dataset_metadata)
        samples: List of sample data

    Returns:
        list[STTSample]: Created samples
    """
    logger.info(
        f"[create_stt_samples] Creating STT samples | "
        f"dataset_id: {dataset.id}, sample_count: {len(samples)}"
    )

    timestamp = now()
    created_samples = [
        STTSample(
            object_store_url=sample_data.object_store_url,
            ground_truth=sample_data.ground_truth,
            language=dataset.language,
            sample_metadata=_extract_metadata_from_url(sample_data.object_store_url),
            dataset_id=dataset.id,
            organization_id=dataset.organization_id,
            project_id=dataset.project_id,
            inserted_at=timestamp,
            updated_at=timestamp,
        )
        for sample_data in samples
    ]

    session.add_all(created_samples)
    session.flush()
    session.commit()

    logger.info(
        f"[create_stt_samples] STT samples created | "
        f"dataset_id: {dataset.id}, created_count: {len(created_samples)}"
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
    base_filter = (
        EvaluationDataset.organization_id == org_id,
        EvaluationDataset.project_id == project_id,
        EvaluationDataset.type == EvaluationType.STT.value,
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
        STTDatasetPublic(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            type=dataset.type,
            language=dataset.language,
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


def get_samples_by_dataset_id(
    *,
    session: Session,
    dataset_id: int,
    org_id: int,
    project_id: int,
    limit: int = 100,
    offset: int = 0,
) -> list[STTSample]:
    """Get samples for a dataset.

    Args:
        session: Database session
        dataset_id: Dataset ID
        org_id: Organization ID
        project_id: Project ID
        limit: Maximum results to return
        offset: Number of results to skip

    Returns:
        list[STTSample]: Samples
    """
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

    return list(session.exec(statement).all())
