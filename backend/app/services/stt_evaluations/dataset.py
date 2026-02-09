"""Dataset management service for STT evaluations."""

import csv
import io
import logging

from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.storage_utils import (
    generate_timestamped_filename,
    upload_to_object_store,
)
from app.crud.stt_evaluations import (
    create_stt_dataset,
    create_stt_samples,
)
from app.models import EvaluationDataset
from app.models.stt_evaluation import STTSample, STTSampleCreate

logger = logging.getLogger(__name__)


def upload_stt_dataset(
    session: Session,
    name: str,
    samples: list[STTSampleCreate],
    organization_id: int,
    project_id: int,
    description: str | None = None,
    language_id: int | None = None,
) -> tuple[EvaluationDataset, list[STTSample]]:
    """
    Orchestrate STT dataset upload workflow.

    Steps:
    1. Convert samples to CSV format
    2. Upload CSV to object store
    3. Create dataset record in database
    4. Create sample records in database

    Args:
        session: Database session
        name: Dataset name
        samples: List of STT samples to create
        organization_id: Organization ID
        project_id: Project ID
        description: Optional dataset description
        language_id: Optional reference to global.languages table

    Returns:
        Tuple of (created dataset, created samples)
    """
    logger.info(
        f"[upload_stt_dataset] Uploading STT dataset | name={name} | "
        f"sample_count={len(samples)} | org_id={organization_id} | "
        f"project_id={project_id}"
    )

    # Step 1: Convert samples to CSV and upload to object store
    object_store_url = _upload_samples_to_object_store(
        session=session,
        project_id=project_id,
        dataset_name=name,
        samples=samples,
    )

    # Step 2: Calculate metadata
    metadata = {
        "sample_count": len(samples),
        "has_ground_truth_count": sum(1 for s in samples if s.ground_truth),
    }

    # Step 3: Create dataset and samples in a single transaction
    try:
        dataset = create_stt_dataset(
            session=session,
            name=name,
            org_id=organization_id,
            project_id=project_id,
            description=description,
            language_id=language_id,
            object_store_url=object_store_url,
            dataset_metadata=metadata,
        )

        logger.info(
            f"[upload_stt_dataset] Created dataset record | "
            f"id={dataset.id} | name={name}"
        )

        # Step 4: Create sample records
        created_samples = create_stt_samples(
            session=session,
            dataset=dataset,
            samples=samples,
        )

        logger.info(
            f"[upload_stt_dataset] Created sample records | "
            f"dataset_id={dataset.id} | sample_count={len(created_samples)}"
        )

        session.commit()

        return dataset, created_samples

    except Exception:
        session.rollback()
        raise


def _upload_samples_to_object_store(
    session: Session,
    project_id: int,
    dataset_name: str,
    samples: list[STTSampleCreate],
) -> str | None:
    """
    Upload STT samples as CSV to object store.

    Args:
        session: Database session
        project_id: Project ID for storage credentials
        dataset_name: Dataset name for filename
        samples: List of samples to upload

    Returns:
        Object store URL if successful, None otherwise
    """
    try:
        storage = get_cloud_storage(session=session, project_id=project_id)

        # Convert samples to CSV format
        csv_content = _samples_to_csv(samples)

        # Generate filename and upload
        filename = generate_timestamped_filename(dataset_name, "csv")
        object_store_url = upload_to_object_store(
            storage=storage,
            content=csv_content,
            filename=filename,
            subdirectory="stt_datasets",
            content_type="text/csv",
        )

        if object_store_url:
            logger.info(
                f"[_upload_samples_to_object_store] Upload successful | "
                f"url={object_store_url}"
            )
        else:
            logger.info(
                "[_upload_samples_to_object_store] Upload returned None | "
                "continuing without object store storage"
            )

        return object_store_url

    except Exception as e:
        logger.warning(
            f"[_upload_samples_to_object_store] Failed to upload | {e}",
            exc_info=True,
        )
        return None


def _samples_to_csv(samples: list[STTSampleCreate]) -> bytes:
    """
    Convert STT samples to CSV format.

    Args:
        samples: List of samples

    Returns:
        CSV content as bytes
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["file_id", "ground_truth"])
    for sample in samples:
        writer.writerow([sample.file_id, sample.ground_truth or ""])
    return output.getvalue().encode("utf-8")
