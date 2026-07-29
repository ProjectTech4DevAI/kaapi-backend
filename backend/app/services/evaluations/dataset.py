"""Dataset management service for evaluations."""

import logging

from fastapi import HTTPException
from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.exceptions import UpstreamError
from app.crud.evaluations import (
    create_evaluation_dataset,
    upload_csv_to_object_store,
    upload_dataset_to_langfuse,
)
from app.crud.evaluations.dataset import (
    DATASET_META_DUPLICATE_AT_RUNTIME,
    DATASET_META_DUPLICATION_FACTOR,
    DATASET_META_ORIGINAL_ITEMS,
    DATASET_META_TOTAL_ITEMS,
)
from app.models.evaluation import EvaluationDataset
from app.services.evaluations.validators import (
    parse_csv_items,
    sanitize_dataset_name,
)
from app.utils import get_langfuse_client

logger = logging.getLogger(__name__)


def upload_dataset(
    session: Session,
    csv_content: bytes,
    dataset_name: str,
    description: str | None,
    duplication_factor: int,
    organization_id: int,
    project_id: int,
) -> EvaluationDataset:
    """
    Orchestrate dataset upload workflow.

    Steps:
    1. Sanitize dataset name
    2. Parse and validate CSV
    3. Upload to object store
    4. Upload to Langfuse
    5. Store metadata in database

    Args:
        session: Database session
        csv_content: Raw CSV file content
        dataset_name: Name for the dataset
        description: Optional dataset description
        duplication_factor: Number of times to duplicate each item
        organization_id: Organization ID
        project_id: Project ID

    Returns:
        Created EvaluationDataset record

    Raises:
        HTTPException: If upload fails at any step
    """
    # Step 1: Sanitize dataset name for Langfuse compatibility
    original_name = dataset_name
    try:
        dataset_name = sanitize_dataset_name(dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid dataset name: {str(e)}")

    if original_name != dataset_name:
        logger.info(
            f"[upload_dataset] Dataset name sanitized | '{original_name}' -> '{dataset_name}'"
        )

    logger.info(
        f"[upload_dataset] Uploading dataset | dataset={dataset_name} | "
        f"duplication_factor={duplication_factor} | org_id={organization_id} | "
        f"project_id={project_id}"
    )

    # Step 2: Parse CSV and extract items
    original_items = parse_csv_items(csv_content)
    original_items_count = len(original_items)
    total_items_count = original_items_count * duplication_factor

    logger.info(
        f"[upload_dataset] Parsed items from CSV | original={original_items_count} | "
        f"total_with_duplication={total_items_count}"
    )

    # Step 3: Upload to object store (if credentials configured)
    object_store_url = None
    try:
        storage = get_cloud_storage(session=session, project_id=project_id)
        object_store_url = upload_csv_to_object_store(
            storage=storage, csv_content=csv_content, dataset_name=dataset_name
        )
        if object_store_url:
            logger.info(
                f"[upload_dataset] Successfully uploaded CSV to object store | {object_store_url}"
            )
        else:
            logger.info(
                "[upload_dataset] Object store upload returned None | "
                "continuing without object store storage"
            )
    except Exception as e:
        logger.warning(
            f"[upload_dataset] Failed to upload CSV to object store "
            f"(continuing without object store) | {e}",
            exc_info=True,
        )
        object_store_url = None

    # Step 4: Upload to Langfuse
    langfuse_dataset_id = None
    try:
        langfuse = get_langfuse_client(
            session=session,
            org_id=organization_id,
            project_id=project_id,
        )

        langfuse_dataset_id, _ = upload_dataset_to_langfuse(
            langfuse=langfuse,
            items=original_items,
            dataset_name=dataset_name,
            duplication_factor=duplication_factor,
        )

        logger.info(
            f"[upload_dataset] Successfully uploaded dataset to Langfuse | "
            f"dataset={dataset_name} | id={langfuse_dataset_id}"
        )

    except Exception:
        logger.error(
            f"[upload_dataset] Failed to upload dataset to Langfuse | dataset={dataset_name}",
            exc_info=True,
        )
        raise UpstreamError(
            f"Langfuse could not accept dataset '{dataset_name}'. Retry shortly.",
            provider="langfuse",
        )

    # Step 5: Store metadata in database
    metadata = {
        "original_items_count": original_items_count,
        "total_items_count": total_items_count,
        "duplication_factor": duplication_factor,
    }

    dataset = create_evaluation_dataset(
        session=session,
        name=dataset_name,
        description=description,
        dataset_metadata=metadata,
        object_store_url=object_store_url,
        langfuse_dataset_id=langfuse_dataset_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    logger.info(
        f"[upload_dataset] Successfully created dataset record in database | "
        f"id={dataset.id} | name={dataset_name}"
    )

    return dataset


def upload_dataset_v2(
    *,
    session: Session,
    csv_content: bytes,
    dataset_name: str,
    description: str | None,
    duplication_factor: int,
    organization_id: int,
    project_id: int,
) -> EvaluationDataset:
    """Uploads a v2 dataset without Langfuse. Stores the original CSV in S3, records
    `duplication_factor` for run-time expansion, and requires a successful S3 upload.
    """
    original_name = dataset_name
    try:
        dataset_name = sanitize_dataset_name(dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid dataset name: {str(e)}")

    if original_name != dataset_name:
        logger.info(
            f"[upload_dataset_v2] Dataset name sanitized | "
            f"'{original_name}' -> '{dataset_name}'"
        )

    original_items = parse_csv_items(csv_content)
    original_items_count = len(original_items)
    total_items_count = original_items_count * duplication_factor

    storage = get_cloud_storage(session=session, project_id=project_id)
    object_store_url = upload_csv_to_object_store(
        storage=storage, csv_content=csv_content, dataset_name=dataset_name
    )
    if not object_store_url:
        # No Langfuse fallback in v2: the run loads items from this CSV, so a
        # missing object-store URL leaves the dataset unrunnable.
        raise HTTPException(
            status_code=500,
            detail="Failed to store dataset CSV in object store",
        )

    metadata = {
        DATASET_META_ORIGINAL_ITEMS: original_items_count,
        DATASET_META_TOTAL_ITEMS: total_items_count,
        DATASET_META_DUPLICATION_FACTOR: duplication_factor,
        DATASET_META_DUPLICATE_AT_RUNTIME: True,
    }

    dataset = create_evaluation_dataset(
        session=session,
        name=dataset_name,
        description=description,
        dataset_metadata=metadata,
        object_store_url=object_store_url,
        langfuse_dataset_id=None,
        organization_id=organization_id,
        project_id=project_id,
    )

    return dataset
