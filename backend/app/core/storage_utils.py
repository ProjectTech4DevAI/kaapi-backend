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
from typing import Literal

logger = logging.getLogger(__name__)


def upload_csv_to_object_store(
    storage: CloudStorage,
    csv_content: bytes,
    filename: str,
    subdirectory: str = "datasets",
) -> str | None:
    """
    Upload CSV content to object store.

    Args:
        storage: CloudStorage instance
        csv_content: Raw CSV content as bytes
        filename: Name of the file (can include timestamp)
        subdirectory: Subdirectory path in object store (default: "datasets")

    Returns:
        Object store URL as string if successful, None if failed

    Note:
        This function handles errors gracefully and returns None on failure.
        Callers should continue without object store URL when this returns None.
    """
    logger.info(
        f"[upload_csv_to_object_store] Preparing to upload '{filename}' | "
        f"size={len(csv_content)} bytes, subdirectory='{subdirectory}'"
    )

    try:
        # Create file path
        file_path = Path(subdirectory) / filename

        # Create a mock UploadFile-like object for the storage put method
        class CSVFile:
            def __init__(self, content: bytes):
                self.file = io.BytesIO(content)
                self.content_type = "text/csv"

        csv_file = CSVFile(csv_content)

        # Upload to object store
        destination = storage.put(source=csv_file, file_path=file_path)
        object_store_url = str(destination)

        logger.info(
            f"[upload_csv_to_object_store] Upload successful | "
            f"filename='{filename}', url='{object_store_url}'"
        )
        return object_store_url

    except CloudStorageError as e:
        logger.warning(
            f"[upload_csv_to_object_store] Upload failed for '{filename}': {e}. "
            "Continuing without object store storage."
        )
        return None
    except Exception as e:
        logger.warning(
            f"[upload_csv_to_object_store] Unexpected error uploading '{filename}': {e}. "
            "Continuing without object store storage.",
            exc_info=True,
        )
        return None


def upload_jsonl_to_object_store(
    storage: CloudStorage,
    results: list[dict],
    filename: str,
    subdirectory: str,
    format: Literal["json", "jsonl"] = "jsonl",
) -> str | None:
    """
    Upload JSONL (JSON Lines) content to object store.

    Args:
        storage: CloudStorage instance
        results: List of dictionaries to be converted to JSONL
        filename: Name of the file
        subdirectory: Subdirectory path in object store (e.g., "evaluation/batch-123")

    Returns:
        Object store URL as string if successful, None if failed

    Note:
        This function handles errors gracefully and returns None on failure.
        Callers should continue without object store URL when this returns None.
    """
    logger.info(
        f"[upload_jsonl_to_object_store] Preparing to upload '{filename}' | "
        f"items={len(results)}, subdirectory='{subdirectory}'"
    )

    try:
        # Create file path
        file_path = Path(subdirectory) / filename

        if format == "jsonl":
            jsonl_content = (
                "\n".join(json.dumps(result, ensure_ascii=False) for result in results)
                + "\n"
            )
            content_type = {"content-type": "application/jsonl"}
        else:
            jsonl_content = json.dumps(results, ensure_ascii=False)
            content_type = {"content-type": "application/json"}

        content_bytes = jsonl_content.encode("utf-8")

        # Create UploadFile-like object
        headers = Headers(content_type)
        upload_file = UploadFile(
            filename=filename,
            file=BytesIO(content_bytes),
            headers=headers,
        )

        # Upload to object store
        destination = storage.put(source=upload_file, file_path=file_path)
        object_store_url = str(destination)

        logger.info(
            f"[upload_jsonl_to_object_store] Upload successful | "
            f"filename='{filename}', url='{object_store_url}', "
            f"size={len(content_bytes)} bytes"
        )
        return object_store_url

    except CloudStorageError as e:
        logger.warning(
            f"[upload_jsonl_to_object_store] Upload failed for '{filename}': {e}. "
            "Continuing without object store storage."
        )
        return None
    except Exception as e:
        logger.warning(
            f"[upload_jsonl_to_object_store] Unexpected error uploading '{filename}': {e}. "
            "Continuing without object store storage.",
            exc_info=True,
        )
        return None


def load_json_from_object_store(storage: CloudStorage, url: str) -> list | dict | None:
    logger.info(f"[load_json_from_object_store] Loading JSON from '{url}")
    try:
        body = storage.stream(url)
        content = body.read()

        data = json.loads(content.decode("utf-8"))

        logger.info(
            f"[load_json_from_object_store] Download successful | "
            f"url='{url}', size={len(content)} bytes"
        )
        return data
    except CloudStorageError as e:
        logger.warning(
            f"[load_json_from_object_store] failed to load JSON from '{url}': {e}",
        )
        return None
    except json.JSONDecodeError as e:
        logger.warning(
            f"[load_json_from_object_store] JSON decode error loading JSON from '{url}': {e}",
        )
        return None
    except Exception as e:
        logger.warning(
            f"[load_json_from_object_store] unexpected error loading JSON from '{url}': {e}",
            exc_info=True,
        )
        return None


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
