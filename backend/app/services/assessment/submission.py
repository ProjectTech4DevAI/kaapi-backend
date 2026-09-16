"""Uploaded submission files for assessments (CSV + XLSX).

Stored as-is: no column validation, no format conversion. The row count is computed at
upload so nothing has to re-read the file to learn how many rows it holds.
"""

import csv
import io
import logging
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.storage_utils import upload_jsonl_to_object_store, upload_to_object_store
from app.crud.assessment.submission import create_submission, get_submission_by_name
from app.models.assessment import AssessmentSubmission
from app.services.assessment.validators import (
    SUBMISSION_FILENAME,
    clean_sheet,
    file_extension_of,
    parse_rows,
    submission_prefix,
)
from app.services.evaluations.validators import sanitize_dataset_name

logger = logging.getLogger(__name__)

try:
    from openpyxl.utils.exceptions import InvalidFileException
except Exception:  # pragma: no cover - openpyxl is expected in runtime deps

    class InvalidFileException(Exception):
        pass


_MIME_TYPES = {
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


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
            subdirectory=submission_prefix(submission_id),
            content_type=content_type,
        )
    except Exception as e:
        logger.warning(
            f"[_upload_file_to_object_store] Failed to upload | {e}",
            exc_info=True,
        )
        return None


def _store_parsed_rows(
    session: Session,
    project_id: int,
    submission_id: UUID,
    rows: list[dict[str, str]],
) -> None:
    """Keep the parsed rows beside the file so no run has to parse it again.

    Best effort: a failure here only costs a later reader one re-parse.
    """
    url = upload_jsonl_to_object_store(
        storage=get_cloud_storage(session=session, project_id=project_id),
        results=rows,
        filename=SUBMISSION_FILENAME,
        subdirectory=submission_prefix(submission_id),
    )
    if not url:
        logger.warning(
            "[_store_parsed_rows] Rows not stored, runs will parse the file | "
            "submission_id=%s",
            submission_id,
        )


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
        rows = parse_rows(file_content, file_ext)
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

    row_count = len(rows)
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

    _store_parsed_rows(session, project_id, submission_id, rows)

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
