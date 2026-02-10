"""CRUD operations for file records."""

import logging

from sqlmodel import Session, select

from app.core.util import now
from app.models.file import File, FileType

logger = logging.getLogger(__name__)


def create_file(
    *,
    session: Session,
    object_store_url: str,
    filename: str,
    size_bytes: int,
    content_type: str,
    file_type: str,
    organization_id: int,
    project_id: int,
) -> File:
    """Create a new file record.

    Args:
        session: Database session
        object_store_url: S3 URL where the file is stored
        filename: Original filename
        size_bytes: File size in bytes
        content_type: MIME type of the file
        file_type: Type of file (audio, document, image, other)
        organization_id: Organization ID
        project_id: Project ID

    Returns:
        File: Created file record
    """
    logger.info(
        f"[create_file] Creating file record | "
        f"filename: {filename}, file_type: {file_type}, "
        f"org_id: {organization_id}, project_id: {project_id}"
    )

    timestamp = now()
    file = File(
        object_store_url=object_store_url,
        filename=filename,
        size_bytes=size_bytes,
        content_type=content_type,
        file_type=file_type,
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=timestamp,
        updated_at=timestamp,
    )

    session.add(file)
    session.commit()
    session.refresh(file)

    logger.info(
        f"[create_file] File record created | "
        f"file_id: {file.id}, filename: {filename}"
    )

    return file


def get_file_by_id(
    *,
    session: Session,
    file_id: int,
    organization_id: int,
    project_id: int,
) -> File | None:
    """Get a file record by ID.

    Args:
        session: Database session
        file_id: File ID
        organization_id: Organization ID
        project_id: Project ID

    Returns:
        File | None: File record if found
    """
    statement = select(File).where(
        File.id == file_id,
        File.organization_id == organization_id,
        File.project_id == project_id,
    )

    return session.exec(statement).one_or_none()


def get_files_by_ids(
    *,
    session: Session,
    file_ids: list[int],
    organization_id: int,
    project_id: int,
) -> list[File]:
    """Get multiple file records by IDs.

    Args:
        session: Database session
        file_ids: List of file IDs
        organization_id: Organization ID
        project_id: Project ID

    Returns:
        list[File]: List of file records found
    """
    if not file_ids:
        return []

    statement = select(File).where(
        File.id.in_(file_ids),
        File.organization_id == organization_id,
        File.project_id == project_id,
    )

    return list(session.exec(statement).all())
