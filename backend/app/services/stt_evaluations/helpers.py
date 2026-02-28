"""Helper functions for building audio file response schemas with signed URLs."""

from app.models.file import File, FilePublic


def build_file_schema(
    *,
    file: File,
    include_url: bool,
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
    if include_url and storage:
        schema.signed_url = storage.get_signed_url(file.object_store_url)
    return schema
