"""Dataset management service for TTS evaluations."""

import csv
import io
import logging
from typing import Any

from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.storage_utils import (
    generate_timestamped_filename,
    upload_to_object_store,
)
from app.crud.tts_evaluations import create_tts_dataset
from app.models import EvaluationDataset
from app.models.tts_evaluation import TTSSampleCreate

logger = logging.getLogger(__name__)


def upload_tts_dataset(
    session: Session,
    name: str,
    samples: list[TTSSampleCreate],
    organization_id: int,
    project_id: int,
    description: str | None = None,
    language_id: int | None = None,
) -> EvaluationDataset:
    """Orchestrate TTS dataset upload workflow.

    Steps:
    1. Convert samples to CSV format
    2. Upload CSV to object store
    3. Create dataset record in database

    Args:
        session: Database session
        name: Dataset name
        samples: List of TTS text samples
        organization_id: Organization ID
        project_id: Project ID
        description: Optional dataset description
        language_id: Optional reference to global.languages table

    Returns:
        Created dataset record
    """
    logger.info(
        f"[upload_tts_dataset] Uploading TTS dataset | name={name} | "
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
    metadata: dict[str, Any] = {
        "sample_count": len(samples),
    }

    # Step 3: Create dataset record
    try:
        dataset = create_tts_dataset(
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
            f"[upload_tts_dataset] Created dataset record | "
            f"id={dataset.id} | name={name}"
        )

        session.commit()

        return dataset

    except Exception:
        session.rollback()
        raise


def _upload_samples_to_object_store(
    session: Session,
    project_id: int,
    dataset_name: str,
    samples: list[TTSSampleCreate],
) -> str | None:
    """Upload TTS samples as CSV to object store.

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

        csv_content = _samples_to_csv(samples)

        filename = generate_timestamped_filename(dataset_name, "csv")
        object_store_url = upload_to_object_store(
            storage=storage,
            content=csv_content,
            filename=filename,
            subdirectory="tts_datasets",
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


def _samples_to_csv(samples: list[TTSSampleCreate]) -> bytes:
    """Convert TTS samples to CSV format.

    Args:
        samples: List of TTS samples

    Returns:
        CSV content as bytes
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text"])
    for sample in samples:
        writer.writerow([sample.text])
    return output.getvalue().encode("utf-8")


def parse_tts_samples_from_csv(csv_content: bytes) -> list[dict[str, Any]]:
    """Parse TTS samples from CSV content.

    Args:
        csv_content: CSV file content as bytes

    Returns:
        List of dicts with {index, text} for each sample
    """
    reader = csv.DictReader(io.StringIO(csv_content.decode("utf-8")))
    samples = []
    for i, row in enumerate(reader):
        text = row.get("text", "").strip()
        if text:
            samples.append({"index": i, "text": text})
    return samples
