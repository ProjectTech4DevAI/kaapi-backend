"""Validation utilities for evaluation datasets."""

import csv
import io
import logging
import re
from pathlib import Path

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 1024 * 1024  # 1 MB
ALLOWED_EXTENSIONS = {".csv"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/csv",
    "text/plain",
}
DEFAULT_CATEGORY = "Other"


def sanitize_dataset_name(name: str) -> str:
    """
    Sanitize dataset name for Langfuse compatibility.

    Langfuse has issues with spaces and special characters in dataset names.
    This function ensures the name can be both created and fetched.

    Rules:
    - Replace spaces with underscores
    - Replace hyphens with underscores
    - Keep only alphanumeric characters and underscores
    - Convert to lowercase for consistency
    - Remove leading/trailing underscores
    - Collapse multiple consecutive underscores into one

    Args:
        name: Original dataset name

    Returns:
        Sanitized dataset name safe for Langfuse

    Examples:
        "testing 0001" -> "testing_0001"
        "My Dataset!" -> "my_dataset"
        "Test--Data__Set" -> "test_data_set"
    """
    sanitized = name.lower()

    # Replace spaces and hyphens with underscores
    sanitized = sanitized.replace(" ", "_").replace("-", "_")

    # Keep only alphanumeric characters and underscores
    sanitized = re.sub(r"[^a-z0-9_]", "", sanitized)

    # Collapse multiple underscores into one
    sanitized = re.sub(r"_+", "_", sanitized)

    sanitized = sanitized.strip("_")

    if not sanitized:
        raise ValueError("Dataset name cannot be empty after sanitization")

    return sanitized


async def validate_csv_file(file: UploadFile) -> bytes:
    """
    Validate CSV file extension, MIME type, and size.

    Args:
        file: The uploaded file

    Returns:
        CSV content as bytes if valid

    Raises:
        HTTPException: If validation fails
    """
    if not file.filename:
        raise HTTPException(
            status_code=422,
            detail="File must have a filename",
        )
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type. Only CSV files are allowed. Got: {file_ext}",
        )

    content_type = file.content_type
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type. Expected CSV, got: {content_type}",
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

    return await file.read()


def parse_csv_items(csv_content: bytes) -> list[dict[str, str]]:
    """
    Parse CSV and extract question/answer/category triples.

    Required columns: `question`, `answer` (case-insensitive).
    Optional column: `category` (case-insensitive) — free-text label used for
    per-category analytics. Missing/blank values default to `"Other"` so old
    CSVs that predate this column continue to work.

    Raises:
        HTTPException: If CSV is invalid or empty
    """
    try:
        csv_text = csv_content.decode("utf-8")
        csv_reader = csv.DictReader(io.StringIO(csv_text))

        if not csv_reader.fieldnames:
            raise HTTPException(status_code=422, detail="CSV file has no headers")

        # Normalize headers for case-insensitive matching
        clean_headers = {
            field.strip().lower(): field for field in csv_reader.fieldnames
        }
        present = set(clean_headers.keys())
        required = {"question", "answer"}
        allowed = required | {"category"}

        missing = required - present
        unexpected = present - allowed
        if missing or unexpected:
            parts = []
            if missing:
                parts.append(f"Missing: {sorted(missing)}")
            if unexpected:
                parts.append(f"Unexpected: {sorted(unexpected)}")
            raise HTTPException(
                status_code=422,
                detail=(
                    "CSV must contain 'question' and 'answer' columns. "
                    "'category' is optional. "
                    f"{'. '.join(parts)}. Found columns: {csv_reader.fieldnames}"
                ),
            )

        question_col = clean_headers["question"]
        answer_col = clean_headers["answer"]
        category_col = clean_headers.get("category")

        items = []
        for row in csv_reader:
            question = row.get(question_col, "").strip()
            answer = row.get(answer_col, "").strip()
            if not (question and answer):
                continue
            raw_category = (
                (row.get(category_col, "") or "").strip() if category_col else ""
            )
            category = raw_category.title() if raw_category else DEFAULT_CATEGORY
            items.append({"question": question, "answer": answer, "category": category})

        if not items:
            raise HTTPException(
                status_code=422, detail="No valid items found in CSV file"
            )

        return items

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"[parse_csv_items] Failed to parse CSV | {e}")
        raise HTTPException(status_code=422, detail=f"Invalid CSV file: {e}")
