"""v2 document upload: pre-signed POST to a pending key, then registration."""

from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi import Path as FastPath

from app.api.deps import AuthContextDep, SessionDep
from app.api.docs.documents.examples import DOCUMENT_UPLOAD_INITIATE_EXAMPLES
from app.api.openapi_docs import request_examples
from app.api.permissions import Permission, require_permission
from app.models import (
    DocumentPublic,
    DocumentUploadInitiateResponse,
    DocumentUploadRequest,
)
from app.services.documents import (
    initiate_document_upload,
    register_uploaded_document,
)
from app.utils import APIResponse, load_description

router = APIRouter(prefix="/documents", tags=["Documents v2"])


@router.post(
    "/uploads",
    description=load_description("documents/v2/initiate.md"),
    response_model=APIResponse[DocumentUploadInitiateResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    openapi_extra=request_examples(DOCUMENT_UPLOAD_INITIATE_EXAMPLES),
)
def create_upload_url(
    session: SessionDep,
    current_user: AuthContextDep,
    request: DocumentUploadRequest,
) -> APIResponse[DocumentUploadInitiateResponse]:
    upload = initiate_document_upload(
        session=session,
        project_id=current_user.project_.id,
        filename=request.filename,
    )

    return APIResponse[DocumentUploadInitiateResponse].success_response(
        upload,
        metadata={
            "next_step": (
                f"Upload the file to the pre-signed S3 URL with the provided fields "
                f"(file part last), then PUT /api/v2/documents/{upload.document_id} to register the document."
            )
        },
    )


@router.put(
    "/{document_id}",
    description=load_description("documents/v2/register.md"),
    status_code=201,
    response_model=APIResponse[DocumentPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def register_document(
    session: SessionDep,
    current_user: AuthContextDep,
    document_id: UUID = FastPath(
        description="Document id issued by the upload session"
    ),
) -> APIResponse[DocumentPublic]:
    document = register_uploaded_document(
        session=session,
        project_id=current_user.project_.id,
        document_id=document_id,
    )

    return APIResponse[DocumentPublic].success_response(
        document,
        metadata={
            "note": "Document registered. The upload URL is spent and cannot be reused."
        },
    )
