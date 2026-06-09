"""Validation utilities for evaluation datasets."""

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
        allowed = required | {"category", "id"}

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
                    "'category' is optional (free-text). "
                    "'id' is optional (when present, every row must have a "
                    "non-empty integer value). "
                    f"{'. '.join(parts)}. Found columns: {csv_reader.fieldnames}"
                ),
            )

        question_col = clean_headers["question"]
        answer_col = clean_headers["answer"]
        category_col = clean_headers.get("category")
        # `id` is case-insensitive (`id` / `Id` / `iD` / `ID` all map here).
        # When the column exists, the per-row value controls the order of
        # the response traces — see fetch_trace_scores_from_langfuse for the
        # sort logic.
        id_col = clean_headers.get("id")

        items = []
        seen_external_ids: set[str] = set()
        # `row_num` reflects the CSV row number as a user would see it in
        # their spreadsheet — row 1 is the header, data starts at row 2.
        for row_num, row in enumerate(csv_reader, start=2):
            question = row.get(question_col, "").strip()
            answer = row.get(answer_col, "").strip()
            if not (question and answer):
                continue
            raw_category = (
                (row.get(category_col, "") or "").strip() if category_col else ""
            )
            category = raw_category.title() if raw_category else DEFAULT_CATEGORY

            # `external_id`: user-provided ordering key. When the `id` column
            # exists, every row MUST have a valid integer value — otherwise
            # the entire upload is rejected with a clear 422 below. This
            # keeps response ordering deterministic and matches the spec
            # that IDs are simple numerics (1, 2, 3, ...). When the column
            # is absent, traces fall back to `question_id` order — the
            # legacy behaviour for datasets uploaded before this feature.
            external_id: str | None = None
            if id_col is not None:
                raw_external_id = (row.get(id_col, "") or "").strip()
                if not raw_external_id:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Row {row_num}: 'id' column is present but the "
                            f"value is missing. When the id column is provided, "
                            f"every row must have a non-empty integer id."
                        ),
                    )
                try:
                    int(raw_external_id)
                except ValueError:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Row {row_num}: id={raw_external_id!r} is not a "
                            f"valid integer. Only integer ids are allowed "
                            f"(e.g. 1, 2, 3, -1). Decimals, trailing "
                            f"punctuation, and letters are not accepted."
                        ),
                    )
                external_id = raw_external_id
                # Surface accidental duplicates so the eval doesn't silently
                # ship two questions with the same ordering key.
                if external_id in seen_external_ids:
                    logger.warning(
                        f"[parse_csv_items] Duplicate id={external_id!r} in CSV; "
                        f"the response order between duplicates is undefined"
                    )
                seen_external_ids.add(external_id)

            items.append(
                {
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "external_id": external_id,
                }
            )

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
