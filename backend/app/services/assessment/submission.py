"""Uploaded submission files for assessments (CSV + XLSX).

Stored as-is: no column validation, no format conversion. The row count is computed at
upload so nothing has to re-read the file to learn how many rows it holds.
"""

import csv
import io
import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.storage_utils import upload_to_object_store
from app.crud.assessment.submission import create_submission, get_submission_by_name
from app.models.assessment import AssessmentSubmission
from app.services.assessment.utils.sheets import clean_sheet
from app.services.evaluations.validators import sanitize_dataset_name

logger = logging.getLogger(__name__)

try:
    from openpyxl.utils.exceptions import InvalidFileException
except Exception:  # pragma: no cover - openpyxl is expected in runtime deps

    class InvalidFileException(Exception):
        pass


SUBMISSIONS_SUBDIRECTORY = "assessment/submissions"

_MIME_TYPES = {
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def file_extension_of(object_store_url: str) -> str:
    """Format of a stored submission, read off its key (no column holds it)."""
    return Path(object_store_url).suffix.lower()


def _upload_file_to_object_store(
    session: Session,
    project_id: int,
    file_content: bytes,
    file_ext: str,
    submission_name: str,
    submission_id: UUID,
) -> str | None:
    """Upload the raw file as-is under a key no other submission can share."""
    filename = f"{submission_name}.{file_ext.lstrip('.')}"
    content_type = _MIME_TYPES.get(file_ext, "application/octet-stream")

    try:
        storage = get_cloud_storage(session=session, project_id=project_id)
        return upload_to_object_store(
            storage=storage,
            content=file_content,
            filename=filename,
            subdirectory=f"{SUBMISSIONS_SUBDIRECTORY}/{submission_id}",
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
    wb = None
    try:
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            return 0

        rows_iter = ws.iter_rows(values_only=True)
        header = next(rows_iter, None)
        if header is None:
            return 0

        return len(clean_sheet(header, rows_iter)[1])
    except InvalidFileException as e:
        logger.warning("[_count_excel_rows] Invalid XLSX file content: %s", e)
        raise
    except Exception as e:
        logger.warning(
            "[_count_excel_rows] Failed to count rows | %s", e, exc_info=True
        )
        raise ValueError("Failed to parse XLSX file") from e
    finally:
        if wb is not None:
            wb.close()


def _count_rows(content: bytes, file_ext: str) -> int:
    """Count data rows in a file (CSV or XLSX), excluding the header."""
    if file_ext == ".xls":
        raise ValueError(
            "Legacy Excel format (.xls) is not supported. Please upload .xlsx or .csv."
        )
    if file_ext == ".xlsx":
        return _count_excel_rows(content)
    return _count_csv_rows(content)


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _preview_csv(content: bytes, limit: int) -> tuple[list[str], list[list[str]]]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        text = content.decode("utf-8", errors="replace")

    reader = csv.reader(io.StringIO(text))
    header = next(reader, None) or []
    headers = [_stringify(cell) for cell in header]

    rows: list[list[str]] = []
    for row in reader:
        if not any(cell.strip() for cell in row):
            continue
        rows.append([_stringify(cell) for cell in row])
        if len(rows) >= limit:
            break
    return headers, rows


def _preview_excel(content: bytes, limit: int) -> tuple[list[str], list[list[str]]]:
    import openpyxl

    wb = None
    try:
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            return [], []

        rows_iter = ws.iter_rows(values_only=True)
        headers, rows = clean_sheet(next(rows_iter, None) or (), rows_iter)
        return headers, rows[:limit]
    finally:
        if wb is not None:
            wb.close()


def preview_submission(
    session: Session,
    submission: AssessmentSubmission,
    project_id: int,
    limit: int,
) -> tuple[list[str], list[list[str]]]:
    """Return the first `limit` data rows (plus header) of a submission file."""
    if not submission.object_store_url:
        raise HTTPException(
            status_code=404, detail="Submission has no underlying file to preview."
        )

    file_ext = file_extension_of(submission.object_store_url)
    if file_ext == ".xls":
        raise HTTPException(
            status_code=422,
            detail="Legacy Excel format (.xls) is not supported.",
        )
    if file_ext not in {".csv", ".xlsx"}:
        raise HTTPException(
            status_code=422,
            detail="Unsupported or missing file extension.",
        )

    storage = get_cloud_storage(session=session, project_id=project_id)
    try:
        content = storage.get(submission.object_store_url)
    except Exception as e:
        logger.warning(
            f"[preview_submission] Failed to fetch file | submission_id={submission.id} | {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502, detail="Failed to fetch the submission file from storage."
        ) from e

    try:
        if file_ext == ".xlsx":
            return _preview_excel(content, limit)
        return _preview_csv(content, limit)
    except InvalidFileException as e:
        raise HTTPException(status_code=422, detail="Invalid XLSX file content.") from e
    except Exception as e:
        logger.warning(
            f"[preview_submission] Failed to parse file | submission_id={submission.id} | {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=422, detail="Unable to parse the submission file for preview."
        ) from e


def upload_submission(
    session: Session,
    file_content: bytes,
    file_ext: str,
    submission_name: str,
    description: str | None,
    organization_id: int,
    project_id: int,
) -> AssessmentSubmission:
    """Store an uploaded submission file and record it."""
    original_name = submission_name
    try:
        submission_name = sanitize_dataset_name(submission_name)
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid submission name: {str(e)}"
        )

    if original_name != submission_name:
        logger.info(
            f"[upload_submission] Name sanitized | '{original_name}' -> '{submission_name}'"
        )

    if get_submission_by_name(
        session=session,
        name=submission_name,
        organization_id=organization_id,
        project_id=project_id,
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Submission with name '{submission_name}' already exists in this "
                "organization and project."
            ),
        )

    try:
        row_count = _count_rows(file_content, file_ext)
    except InvalidFileException as e:
        raise HTTPException(
            status_code=422,
            detail="Invalid XLSX file content. Please upload a valid .xlsx file.",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="Unable to parse the file. Please upload a valid CSV or XLSX file.",
        ) from e

    logger.info(
        f"[upload_submission] Uploading | name={submission_name} | "
        f"file_type={file_ext} | rows={row_count} | "
        f"org_id={organization_id} | project_id={project_id}"
    )

    submission_id = uuid4()
    object_store_url = _upload_file_to_object_store(
        session=session,
        project_id=project_id,
        file_content=file_content,
        file_ext=file_ext,
        submission_name=submission_name,
        submission_id=submission_id,
    )
    if not object_store_url:
        logger.error(
            f"[upload_submission] Object store upload failed | name={submission_name} | "
            f"org_id={organization_id} | project_id={project_id}"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to upload the submission file. Please try again.",
        )

    submission = create_submission(
        session=session,
        submission_id=submission_id,
        name=submission_name,
        description=description,
        object_store_url=object_store_url,
        total_items=row_count,
        organization_id=organization_id,
        project_id=project_id,
    )

    logger.info(
        f"[upload_submission] Created record | "
        f"id={submission.id} | name={submission_name} | rows={row_count}"
    )

    return submission
