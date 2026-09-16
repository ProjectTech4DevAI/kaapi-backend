"""Uploaded submission files: upload validation, object-store keys, and parsing to rows.

Every layer imports this, so it stays dependency-free: no DB, no object store, no
other app module.
"""

import csv
import io
import logging
from collections.abc import Iterable
from pathlib import Path
from uuid import UUID

import openpyxl
from fastapi import HTTPException, UploadFile
from openpyxl.utils.exceptions import InvalidFileException

logger = logging.getLogger(__name__)


# ─── UPLOAD VALIDATION ───────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/csv",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


async def validate_dataset_file(file: UploadFile) -> tuple[bytes, str]:
    """Validate an uploaded dataset file (CSV or XLSX).

    Only checks file type and size — does NOT inspect columns.

    Returns:
        Tuple of (file content as bytes, file extension)

    Raises:
        HTTPException: If validation fails
    """
    if not file.filename:
        raise HTTPException(status_code=422, detail="File must have a filename")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext == ".xls":
        raise HTTPException(
            status_code=422,
            detail="Legacy Excel format (.xls) is not supported. Please upload .xlsx or .csv.",
        )
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type. Allowed: CSV, XLSX. Got: {file_ext}",
        )

    content_type = file.content_type
    if content_type not in ALLOWED_MIME_TYPES:
        logger.warning(
            f"[validate_dataset_file] Unexpected content type '{content_type}' "
            f"for extension '{file_ext}', proceeding based on extension"
        )

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB",
        )

    if file_size == 0:
        raise HTTPException(status_code=422, detail="Empty file uploaded")

    content = await file.read()
    return content, file_ext


# ─── STORAGE KEY CONSTANTS ───────────────────────────────────────────────────

SUBMISSION_FILENAME = "submission.jsonl"
ERRORS_FILENAME = "errors.jsonl"

ASSESSMENTS_PREFIX = "assessment"
SUBMISSIONS_PREFIX = "assessment/submissions"


def file_extension_of(object_store_url: str) -> str:
    """Format of a stored submission, read off its key (no column holds it)."""
    return Path(object_store_url).suffix.lower()


def assessment_prefix(assessment_id: UUID) -> str:
    """Holds every file one assessment produces."""
    return f"{ASSESSMENTS_PREFIX}/{assessment_id}"


def stage_batch_prefix(assessment_id: UUID, batch_job_id: int) -> str:
    """Holds one stage's provider dump."""
    return f"{assessment_prefix(assessment_id)}/batch-{batch_job_id}"


def submission_prefix(submission_id: UUID) -> str:
    """Holds an uploaded file and the rows parsed from it."""
    return f"{SUBMISSIONS_PREFIX}/{submission_id}"


def submission_rows_url(object_store_url: str) -> str:
    """The parsed rows sitting beside an uploaded file.

    String surgery, not ``Path``: a posix path collapses the ``s3://`` double slash.
    """
    return f"{object_store_url.rsplit('/', 1)[0]}/{SUBMISSION_FILENAME}"


# ─── SHEET PARSING ───────────────────────────────────────────────────────────


def _cell(value: object) -> str:
    return "" if value is None else str(value).strip()


def clean_sheet(
    header: Iterable[object], rows: Iterable[Iterable[object]]
) -> tuple[list[str], list[list[str]]]:
    """Drop blank rows, then columns with no header or no values; cells come back stripped.

    Sheets pad with formatted-but-empty cells, so 1000 rows x 17 columns with 100 x 10
    filled comes back as exactly 100 x 10. A row left empty by the column drop goes too.
    """
    headers = [_cell(name) for name in header]
    width = len(headers)

    filled: list[list[str]] = []
    for row in rows:
        cells = [_cell(value) for value in row][:width]
        cells += [""] * (width - len(cells))
        if any(cells):
            filled.append(cells)

    keep = [
        idx
        for idx, name in enumerate(headers)
        if name and any(row[idx] for row in filled)
    ]
    rows = [[row[idx] for idx in keep] for row in filled]
    return [headers[idx] for idx in keep], [row for row in rows if any(row)]


def _named_cells(row: dict[str | None, str | None]) -> dict[str, str]:
    """Keep only columns the sheet actually names.

    A spreadsheet's trailing blank columns are padding, not data — naming them
    ``col_<n>`` would surface them as undeclared columns and 422 the run.
    """
    return {
        str(key).strip(): value or ""
        for key, value in row.items()
        if key is not None and str(key).strip()
    }


def parse_csv_rows(content: bytes) -> list[dict[str, str]]:
    """Parse CSV content into list of row dicts."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        text = content.decode("utf-8", errors="replace")

    rows = (_named_cells(row) for row in csv.DictReader(io.StringIO(text)))
    return [row for row in rows if any(value.strip() for value in row.values())]


def parse_excel_rows(content: bytes) -> list[dict[str, str]]:
    """Parse Excel content into row dicts; blank rows and empty columns are dropped."""
    wb = None
    try:
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            return []

        rows_iter = ws.iter_rows(values_only=True)
        header = next(rows_iter, None)
        if header is None:
            return []

        headers, rows = clean_sheet(header, rows_iter)
        return [dict(zip(headers, row, strict=True)) for row in rows]
    except InvalidFileException as e:
        logger.warning("[parse_excel_rows] Invalid XLSX file content: %s", e)
        raise
    except Exception as e:
        logger.warning(
            "[parse_excel_rows] Failed to parse XLSX rows | %s", e, exc_info=True
        )
        raise ValueError("Failed to parse XLSX submission rows") from e
    finally:
        if wb is not None:
            wb.close()


def parse_rows(content: bytes, file_ext: str) -> list[dict[str, str]]:
    """Parse an uploaded submission file's bytes by the format its key names."""
    if file_ext == ".xls":
        raise ValueError(
            "Legacy Excel format (.xls) is not supported. Please upload .xlsx or .csv."
        )
    if file_ext == ".xlsx":
        return parse_excel_rows(content)
    return parse_csv_rows(content)
