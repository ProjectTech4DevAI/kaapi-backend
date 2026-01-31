"""
Shared storage utilities for uploading files to object store.

This module provides common functions for uploading various file types
to cloud object storage, abstracting away provider-specific details.
"""

import io
import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path

from starlette.datastructures import Headers, UploadFile

from app.core.cloud.storage import CloudStorage, CloudStorageError

logger = logging.getLogger(__name__)


def upload_to_object_store(
    storage: CloudStorage,
    content: bytes,
    filename: str,
    subdirectory: str,
    content_type: str = "application/octet-stream",
) -> str | None:
    """
    Upload content to object store.

    This is the generic upload function that handles any content type.
    Use this directly or through convenience wrappers like upload_csv_to_object_store.

    Args:
        storage: CloudStorage instance
        content: Raw content as bytes
        filename: Name of the file
        subdirectory: Subdirectory path in object store (e.g., "datasets", "stt_datasets")
        content_type: MIME type of the content (default: "application/octet-stream")

    Returns:
        Object store URL as string if successful, None if failed

    Note:
        This function handles errors gracefully and returns None on failure.
        Callers should continue without object store URL when this returns None.
    """
    logger.info(
        f"[upload_to_object_store] Preparing to upload '{filename}' | "
        f"size={len(content)} bytes, subdirectory='{subdirectory}', "
        f"content_type='{content_type}'"
    )

    try:
        file_path = Path(subdirectory) / filename

        headers = Headers({"content-type": content_type})
        upload_file = UploadFile(
            filename=filename,
            file=BytesIO(content),
            headers=headers,
        )

        destination = storage.put(source=upload_file, file_path=file_path)
        object_store_url = str(destination)

        logger.info(
            f"[upload_to_object_store] Upload successful | "
            f"filename='{filename}', url='{object_store_url}'"
        )
        return object_store_url

    except CloudStorageError as e:
        logger.warning(
            f"[upload_to_object_store] Upload failed for '{filename}': {e}. "
            "Continuing without object store storage."
        )
        return None
    except Exception as e:
        logger.warning(
            f"[upload_to_object_store] Unexpected error uploading '{filename}': {e}. "
            "Continuing without object store storage.",
            exc_info=True,
        )
        return None


def upload_csv_to_object_store(
    storage: CloudStorage,
    csv_content: bytes,
    filename: str,
    subdirectory: str = "datasets",
) -> str | None:
    """
    Upload CSV content to object store.

    Convenience wrapper around upload_to_object_store for CSV files.

    Args:
        storage: CloudStorage instance
        csv_content: Raw CSV content as bytes
        filename: Name of the file (can include timestamp)
        subdirectory: Subdirectory path in object store (default: "datasets")

    Returns:
        Object store URL as string if successful, None if failed
    """
    return upload_to_object_store(
        storage=storage,
        content=csv_content,
        filename=filename,
        subdirectory=subdirectory,
        content_type="text/csv",
    )


def upload_jsonl_to_object_store(
    storage: CloudStorage,
    results: list[dict],
    filename: str,
    subdirectory: str,
) -> str | None:
    """
    Upload JSONL (JSON Lines) content to object store.

    Convenience wrapper around upload_to_object_store for JSONL files.

    Args:
        storage: CloudStorage instance
        results: List of dictionaries to be converted to JSONL
        filename: Name of the file
        subdirectory: Subdirectory path in object store

    Returns:
        Object store URL as string if successful, None if failed
    """
    jsonl_content = "\n".join([json.dumps(result) for result in results])
    content_bytes = jsonl_content.encode("utf-8")

    return upload_to_object_store(
        storage=storage,
        content=content_bytes,
        filename=filename,
        subdirectory=subdirectory,
        content_type="application/jsonl",
    )


def generate_timestamped_filename(base_name: str, extension: str = "csv") -> str:
    """
    Generate a filename with timestamp.

    Args:
        base_name: Base name for the file (e.g., "dataset_name" or "batch-123")
        extension: File extension without dot (default: "csv")

    Returns:
        Filename with timestamp (e.g., "dataset_name_20250114_153045.csv")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"
