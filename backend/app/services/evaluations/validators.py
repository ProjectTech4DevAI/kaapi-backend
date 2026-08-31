"""Validation utilities for evaluation datasets."""

import codecs
import csv
import io
import logging
import re
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.crud.evaluations.score import DEFAULT_CATEGORY  # noqa: F401

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 1024 * 1024  # 1 MB
ALLOWED_EXTENSIONS = {".csv"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/csv",
    "text/plain",
}

# Checked longest-first: the UTF-32-LE BOM starts with the UTF-16-LE BOM,
# so probing UTF-16 first would misread a UTF-32 file.
BOM_ENCODINGS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# cp1252 before latin-1: it maps 0x80-0x9F to the smart quotes, apostrophes and
# dashes Windows Excel writes there, which latin-1 would turn into C1 controls.
FALLBACK_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


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


def decode_csv_bytes(csv_content: bytes) -> str:
    """Decode uploaded CSV bytes, tolerating the encodings spreadsheet apps emit.

    Handles UTF-8, UTF-8 with a BOM (Excel's "CSV UTF-8" export), UTF-16/UTF-32
    with a BOM (Excel's "Unicode Text" export) and the cp1252 bytes Windows Excel
    writes for its plain "CSV (Comma delimited)" export.

    Raises:
        HTTPException: If the bytes cannot be decoded by any candidate encoding.
    """
    for bom, encoding in BOM_ENCODINGS:
        if csv_content.startswith(bom):
            return csv_content.decode(encoding)

    for encoding in FALLBACK_ENCODINGS:
        try:
            return csv_content.decode(encoding)
        except UnicodeDecodeError:
            continue

    # Unreachable while latin-1 (which decodes any byte sequence) closes the ladder;
    # kept so the function stays total if FALLBACK_ENCODINGS loses its catch-all codec.
    logger.warning("[decode_csv_bytes] Failed to decode CSV with any known encoding")
    raise HTTPException(
        status_code=422,
        detail="Unable to read the CSV file. Please re-save it as UTF-8 CSV.",
    )


def parse_csv_items(csv_content: bytes) -> list[dict[str, str]]:
    """
    Parse CSV and extract question/answer/category triples.

    Required columns: `question`, `answer` (case-insensitive).
    Optional column: `category` (case-insensitive) — free-text label used for
    per-category analytics. When the column is present, blank cells default to
    `"Other"`. When the column is absent entirely, no `category` field is added
    to items at all, so downstream traces and the API response stay clean of a
    category dimension the user didn't opt into.

    Raises:
        HTTPException: If CSV is invalid or empty
    """
    try:
        csv_text = decode_csv_bytes(csv_content)
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
            item: dict[str, str] = {"question": question, "answer": answer}
            if category_col is not None:
                raw_category = (row.get(category_col, "") or "").strip()
                item["category"] = (
                    raw_category.title() if raw_category else DEFAULT_CATEGORY
                )
            items.append(item)

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
