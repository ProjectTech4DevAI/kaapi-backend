"""Helper functions for building audio file response schemas with signed URLs."""

import logging
from typing import Iterable

from app.models.file import File, FilePublic

logger = logging.getLogger(__name__)


def build_file_schema(
    *,
    file: File,
    include_signed_url: bool,
    storage: object | None,
) -> FilePublic:
    """Build a single file schema, optionally attaching a signed URL.

    Args:
        file: The File database model instance
        include_url: Whether to generate and include a signed URL
        storage: Cloud storage instance for generating signed URLs

    Returns:
        FilePublic schema with optional signed_url
    """
    schema = FilePublic.model_validate(file, from_attributes=True)
    if include_signed_url and storage:
        try:
            schema.signed_url = storage.get_signed_url(file.object_store_url)
        except Exception as e:
            logger.warning(
                f"[build_file_schema] Failed to generate signed URL for file {file.id}"
            )
            schema.signed_url = None
    return schema


def build_file_schemas(
    *,
    files: Iterable[File],
    include_signed_url: bool,
    storage: object | None,
) -> list[FilePublic]:
    """Build multiple file schemas efficiently, optionally attaching signed URLs.

    Args:
        files: Iterable of File database model instances
        include_url: Whether to generate and include signed URLs
        storage: Cloud storage instance for generating signed URLs

    Returns:
        List of FilePublic schemas with optional signed_url
    """
    out: list[FilePublic] = []
    for file in files:
        schema = FilePublic.model_validate(file, from_attributes=True)
        if include_signed_url and storage:
            try:
                schema.signed_url = storage.get_signed_url(file.object_store_url)
            except Exception as e:
                logger.warning(
                    f"[build_file_schemas] Failed to generate signed URL for file {file.id}"
                )
                schema.signed_url = None
        out.append(schema)
    return out
