"""v2 document upload: pre-signed PUT to storage, then registration. No transformation."""

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.cloud import get_cloud_storage
from app.core.cloud.storage import SimpleStorageName
from app.crud import DocumentCrud
from app.models import (
    Document,
    DocumentPublic,
    DocumentRegisterRequest,
    DocumentUploadResponse,
    DocumentUploadURLRequest,
    DocumentUploadURLResponse,
)
from app.services.documents.helpers import (
    validate_filename_format,
    verify_uploaded_object,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents v2"])

UPLOAD_URL_EXPIRY_SECONDS = 3600


@router.post(
    "/upload-url",
    description=load_description("documents/upload_url_v2.md"),
    response_model=APIResponse[DocumentUploadURLResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_upload_url(
    session: SessionDep,
    current_user: AuthContextDep,
    request: DocumentUploadURLRequest,
) -> APIResponse[DocumentUploadURLResponse]:
    validate_filename_format(request.filename)

    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)
    document_id = uuid4()
    key = Path(storage.storage_path) / str(document_id)
    upload_url = storage.get_signed_upload_url(
        key.as_posix(),
        expires_in=UPLOAD_URL_EXPIRY_SECONDS,
    )

    logger.info(
        f"[create_upload_url] Upload URL issued | "
        f"document_id: {document_id}, project_id: {current_user.project_.id}"
    )

    return APIResponse[DocumentUploadURLResponse].success_response(
        DocumentUploadURLResponse(
            document_id=document_id,
            upload_url=upload_url,
            expires_in=UPLOAD_URL_EXPIRY_SECONDS,
        )
    )


@router.post(
    "",
    description=load_description("documents/register_v2.md"),
    status_code=201,
    response_model=APIResponse[DocumentUploadResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def register_document(
    session: SessionDep,
    current_user: AuthContextDep,
    request: DocumentRegisterRequest,
) -> APIResponse[DocumentUploadResponse]:
    validate_filename_format(request.filename)

    crud = DocumentCrud(session, current_user.project_.id)
    if crud.exists(request.document_id):
        logger.warning(
            f"[register_document] Document already registered | "
            f"document_id: {request.document_id}, project_id: {current_user.project_.id}"
        )
        raise HTTPException(
            status_code=409,
            detail="This document_id is already registered. Request a new upload URL.",
        )

    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)
    key = Path(storage.storage_path) / str(request.document_id)
    object_store_url = str(SimpleStorageName(key.as_posix()))

    file_size_kb = verify_uploaded_object(
        storage=storage,
        object_store_url=object_store_url,
        document_id=request.document_id,
    )

    document = crud.update(
        Document(
            id=request.document_id,
            fname=request.filename,
            file_size_kb=file_size_kb,
            object_store_url=object_store_url,
            project_id=current_user.project_.id,
        )
    )

    document_schema = DocumentPublic.model_validate(document, from_attributes=True)
    document_schema.signed_url = storage.get_signed_url(document.object_store_url)

    logger.info(
        f"[register_document] Document registered | "
        f"document_id: {document.id}, project_id: {current_user.project_.id}, size_kb: {file_size_kb}"
    )

    return APIResponse[DocumentUploadResponse].success_response(
        DocumentUploadResponse(**document_schema.model_dump())
    )
