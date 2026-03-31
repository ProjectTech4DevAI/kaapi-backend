"""Validation utilities for assessment dataset file uploads (CSV + Excel).

Only validates file type and size — no column requirements.
"""

import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/csv",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


async def validate_dataset_file(file: UploadFile) -> tuple[bytes, str]:
    """Validate an uploaded dataset file (CSV or Excel).

    Only checks file type and size — does NOT inspect columns.

    Returns:
        Tuple of (file content as bytes, file extension)

    Raises:
        HTTPException: If validation fails
    """
    if not file.filename:
        raise HTTPException(status_code=422, detail="File must have a filename")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type. Allowed: CSV, XLSX, XLS. Got: {file_ext}",
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
