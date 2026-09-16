"""v2 upload policy: promote what the client uploaded into a document row."""

from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.cloud.storage import ObjectNotFoundError
from app.crud import DocumentCrud
from app.models import Document
from app.services.doctransform.registry import get_file_format

DUPLICATE_DOCUMENT_DETAIL = (
    "This document_id is already registered. Request a new upload URL."
)
MISSING_UPLOAD_DETAIL = (
    "No uploaded file found for this document_id. Upload the file to the "
    "pre-signed URL before registering it."
)


def validate_filename_format(filename: str) -> str:
    """Resolve the document format from the extension; HTTPException(400) if unsupported."""
    try:
        return get_file_format(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def register_uploaded_document(
    *,
    session: Session,
    project_id: int,
    document_id: UUID,
) -> Document:
    """Promote an uploaded object into a document row, moving it to its final key.

    The filename comes from the object's own signed metadata, not the request, so
    it cannot differ from what the upload ticket was issued for. Size is already
    capped by the ticket, so the copied object is measured only to record its size.
    """
    document_crud = DocumentCrud(session, project_id)
    if document_crud.exists(document_id):
        raise HTTPException(status_code=409, detail=DUPLICATE_DOCUMENT_DETAIL)

    storage = get_cloud_storage(session=session, project_id=project_id)
    pending_url = str(storage.url_for(Path(str(document_id)), is_pending=True))

    # Copy first, then read the frozen final key: the pending object stays writable
    # through its ticket, so measuring it directly would race the copy.
    try:
        object_store_url = storage.copy(pending_url, Path(str(document_id)))
    except ObjectNotFoundError:
        raise HTTPException(status_code=400, detail=MISSING_UPLOAD_DETAIL)

    stored = storage.head(str(object_store_url))
    filename = stored.filename or str(document_id)

    try:
        document = document_crud.update(
            Document(
                id=document_id,
                fname=filename,
                file_size_kb=stored.size_kb,
                object_store_url=str(object_store_url),
                project_id=project_id,
            )
        )
    except IntegrityError:
        # Two completions can both clear exists(); the PK decides, and the object is the winner's.
        session.rollback()
        raise HTTPException(status_code=409, detail=DUPLICATE_DOCUMENT_DETAIL)

    # Only once the row is committed: a failed insert leaves the upload retryable,
    # and an abandoned pending object expires on its own.
    storage.delete(pending_url)
    return document
