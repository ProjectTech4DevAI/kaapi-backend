"""v2 document upload: pre-signed PUT to a pending key, then registration."""

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends
from fastapi import Path as FastPath

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.cloud import get_cloud_storage
from app.models import (
    DocumentPublic,
    DocumentUploadInitiateResponse,
    DocumentUploadRequest,
)
from app.services.documents.registration import (
    register_uploaded_document,
    validate_filename_format,
)
from app.utils import APIResponse, load_description

router = APIRouter(prefix="/documents", tags=["Documents v2"])

UPLOAD_URL_EXPIRY_SECONDS = 3600


@router.post(
    "/uploads",
    description=load_description("documents/initiate_v2.md"),
    response_model=APIResponse[DocumentUploadInitiateResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_upload_url(
    session: SessionDep,
    current_user: AuthContextDep,
    request: DocumentUploadRequest,
) -> APIResponse[DocumentUploadInitiateResponse]:
    validate_filename_format(request.filename)

    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)
    document_id = uuid4()
    extension = Path(request.filename).suffix.lower()
    signed = storage.get_signed_upload_url(
        Path(f"{document_id}{extension}"),
        expires_in=UPLOAD_URL_EXPIRY_SECONDS,
    )

    return APIResponse[DocumentUploadInitiateResponse].success_response(
        DocumentUploadInitiateResponse(
            document_id=document_id,
            upload_signed_url=signed.url,
            expires_in=signed.expires_in,
        )
    )


@router.put(
    "/{document_id}",
    description=load_description("documents/register_v2.md"),
    status_code=201,
    response_model=APIResponse[DocumentPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def register_document(
    session: SessionDep,
    current_user: AuthContextDep,
    request: DocumentUploadRequest,
    document_id: UUID = FastPath(
        description="Document id issued by the upload session"
    ),
) -> APIResponse[DocumentPublic]:
    document = register_uploaded_document(
        session=session,
        project_id=current_user.project_.id,
        document_id=document_id,
        filename=request.filename,
    )

    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)
    document_schema = DocumentPublic.model_validate(document, from_attributes=True)
    document_schema.signed_url = storage.get_signed_url(document.object_store_url)

    return APIResponse[DocumentPublic].success_response(document_schema)
