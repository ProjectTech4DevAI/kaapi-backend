"""Dataset management service for assessment evaluations (CSV + Excel).

Upload stores files directly to object store as-is (no column validation,
no format conversion). Row count is computed for metadata.
"""

import csv
import io
import logging

from fastapi import HTTPException
from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.storage_utils import generate_timestamped_filename, upload_to_object_store
from app.crud.evaluations import create_evaluation_dataset
from app.models.evaluation import EvaluationDataset
from app.services.evaluations.validators import sanitize_dataset_name

logger = logging.getLogger(__name__)

_MIME_TYPES = {
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
}


def _upload_file_to_object_store(
    session: Session,
    project_id: int,
    file_content: bytes,
    file_ext: str,
    dataset_name: str,
) -> str | None:
    """Upload the raw file to object store, preserving original format."""
    extension = file_ext.lstrip(".")
    filename = generate_timestamped_filename(dataset_name, extension=extension)
    content_type = _MIME_TYPES.get(file_ext, "application/octet-stream")

    try:
        storage = get_cloud_storage(session=session, project_id=project_id)
        return upload_to_object_store(
            storage=storage,
            content=file_content,
            filename=filename,
            subdirectory="datasets",
            content_type=content_type,
        )
    except Exception as e:
        logger.warning(
            f"[_upload_file_to_object_store] Failed to upload | {e}",
            exc_info=True,
        )
        return None


def _count_csv_rows(content: bytes) -> int:
    """Count data rows in a CSV file (excluding header)."""
    try:
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = content.decode(encoding)
                break
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            text = content.decode("utf-8", errors="replace")

        reader = csv.reader(io.StringIO(text))
        next(reader, None)
        return sum(1 for row in reader if any(cell.strip() for cell in row))
    except Exception as e:
        logger.warning(f"[_count_csv_rows] Failed to count rows | {e}")
        return 0


def _count_excel_rows(content: bytes) -> int:
    """Count data rows in an Excel file (excluding header)."""
    try:
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            return 0

        rows_iter = ws.iter_rows(values_only=True)
        header = next(rows_iter, None)
        if header is None:
            wb.close()
            return 0

        count = sum(
            1 for row in rows_iter if row and any(cell is not None for cell in row)
        )
        wb.close()
        return count
    except Exception as e:
        logger.warning(f"[_count_excel_rows] Failed to count rows | {e}")
        return 0


def _count_rows(content: bytes, file_ext: str) -> int:
    """Count data rows in a file (CSV or Excel), excluding the header."""
    if file_ext in (".xlsx", ".xls"):
        return _count_excel_rows(content)
    return _count_csv_rows(content)


def upload_dataset(
    session: Session,
    file_content: bytes,
    file_ext: str,
    dataset_name: str,
    description: str | None,
    organization_id: int,
    project_id: int,
) -> EvaluationDataset:
    """Upload a dataset file directly to object store and record metadata."""
    original_name = dataset_name
    try:
        dataset_name = sanitize_dataset_name(dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid dataset name: {str(e)}")

    if original_name != dataset_name:
        logger.info(
            f"[upload_dataset] Dataset name sanitized | '{original_name}' -> '{dataset_name}'"
        )

    row_count = _count_rows(file_content, file_ext)

    logger.info(
        f"[upload_dataset] Uploading dataset | dataset={dataset_name} | "
        f"file_type={file_ext} | rows={row_count} | "
        f"org_id={organization_id} | project_id={project_id}"
    )

    object_store_url = _upload_file_to_object_store(
        session=session,
        project_id=project_id,
        file_content=file_content,
        file_ext=file_ext,
        dataset_name=dataset_name,
    )

    metadata = {
        "file_extension": file_ext,
        "file_size_bytes": len(file_content),
        "total_items_count": row_count,
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

    logger.info(
        f"[upload_dataset] Created dataset record | "
        f"id={dataset.id} | name={dataset_name} | rows={row_count}"
    )

    return dataset
