"""v2 upload policy: verify what the client staged, then promote it to a document row."""

from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.cloud.storage import CloudStorage, ObjectNotFoundError
from app.crud import DocumentCrud
from app.models import Document
from app.services.collections.helpers import MAX_DOC_SIZE_MB
from app.services.doctransform.registry import get_file_format

DUPLICATE_DOCUMENT_DETAIL = (
    "This document_id is already registered. Request a new upload URL."
)


def validate_filename_format(filename: str) -> str:
    """Resolve the document format from the extension; HTTPException(400) if unsupported."""
    try:
        return get_file_format(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def verify_pending_object(
    *,
    storage: CloudStorage,
    pending_url: str,
    document_id: UUID,
) -> float:
    """Confirm the pending object exists and fits the size budget; return size in KB."""
    try:
        file_size_kb = storage.get_file_size_kb(pending_url)
    except ObjectNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="No uploaded file found for this document_id. Upload the file to the "
            "pre-signed URL first, and pass the same filename the upload URL was issued "
            "for — its extension determines where the bytes were staged.",
        )

    file_size_mb = file_size_kb / 1024
    if file_size_mb > MAX_DOC_SIZE_MB:
        storage.delete(pending_url)
        raise HTTPException(
            status_code=413,
            detail=f"Document size ({round(file_size_mb, 2)} MB) exceeds the maximum allowed size of {MAX_DOC_SIZE_MB} MB. "
            f"Please upload a smaller file.",
        )

    return file_size_kb


def register_uploaded_document(
    *,
    session: Session,
    project_id: int,
    document_id: UUID,
    filename: str,
) -> Document:
    """Promote a staged upload into a document row, moving the object to its final key."""
    validate_filename_format(filename)

    document_crud = DocumentCrud(session, project_id)
    if document_crud.exists(document_id):
        raise HTTPException(status_code=409, detail=DUPLICATE_DOCUMENT_DETAIL)

    storage = get_cloud_storage(session=session, project_id=project_id)
    extension = Path(filename).suffix.lower()
    pending_url = str(
        storage.url_for(Path(f"{document_id}{extension}"), is_pending=True)
    )

    file_size_kb = verify_pending_object(
        storage=storage,
        pending_url=pending_url,
        document_id=document_id,
    )

    object_store_url = storage.copy(pending_url, Path(str(document_id)))
    storage.delete(pending_url)

    try:
        document = document_crud.update(
            Document(
                id=document_id,
                fname=filename,
                file_size_kb=file_size_kb,
                object_store_url=str(object_store_url),
                project_id=project_id,
            )
        )
    except IntegrityError:
        # Two completions can both clear exists(); the PK decides, and the object is the winner's.
        session.rollback()
        raise HTTPException(status_code=409, detail=DUPLICATE_DOCUMENT_DETAIL)

    return document
