import logging
from pathlib import Path
from typing import Union
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    UploadFile,
)
from fastapi import Path as FastPath

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud import CollectionCrud, DocumentCrud
from app.crud.rag import OpenAIAssistantCrud, OpenAIVectorStoreCrud
from app.models import (
    Document,
    DocumentPublic,
    TransformedDocumentPublic,
    DocumentUploadResponse,
    Message,
    TransformationJobInfo,
    DocTransformationJobPublic,
)
from app.core.cloud import get_cloud_storage
from app.services.collections.helpers import pick_service_for_documennt
from app.services.documents.helpers import (
    schedule_transformation,
    pre_transform_validation,
    build_document_schema,
    build_document_schemas,
)
from app.utils import (
    APIResponse,
    get_openai_client,
    load_description,
    validate_callback_url,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])
doctransformation_callback_router = APIRouter()


@doctransformation_callback_router.post(
    "{$callback_url}",
    name="doctransformation_callback",
)
def doctransformation_callback_notification(
    body: APIResponse[DocTransformationJobPublic],
):
    """
    Callback endpoint specification for document transformation.

    The callback will receive:
    - On success: APIResponse with success=True and data containing DocTransformationJobPublic
    - On failure: APIResponse with success=False and error message
    - metadata field will always be included if provided in the request
    """
    ...


@router.get(
    "/",
    description=load_description("documents/list.md"),
    response_model=APIResponse[list[Union[DocumentPublic, TransformedDocumentPublic]]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_docs(
    session: SessionDep,
    current_user: AuthContextDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, gt=0, le=100),
    include_url: bool = Query(
        False, description="Include a signed URL to access each document"
    ),
):
    crud = DocumentCrud(session, current_user.project_.id)
    documents = crud.read_many(skip, limit)

    storage = (
        get_cloud_storage(session=session, project_id=current_user.project_.id)
        if include_url and documents
        else None
    )

    results = build_document_schemas(
        documents=documents,
        include_url=include_url,
        storage=storage,
    )
    return APIResponse.success_response(results)


@router.post(
    "/",
    description=load_description("documents/upload.md"),
    response_model=APIResponse[DocumentUploadResponse],
    callbacks=doctransformation_callback_router.routes,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def upload_doc(
    session: SessionDep,
    current_user: AuthContextDep,
    src: UploadFile = File(...),
    target_format: str
    | None = Form(
        None,
        description="Desired output format for the uploaded document (e.g., pdf, docx, txt).",
    ),
    transformer: str
    | None = Form(
        None, description="Name of the transformer to apply when converting."
    ),
    callback_url: str
    | None = Form(None, description="URL to call to report doc transformation status"),
):
    if callback_url:
        validate_callback_url(callback_url)

    source_format, actual_transformer = pre_transform_validation(
        src_filename=src.filename,
        target_format=target_format,
        transformer=transformer,
    )

    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)
    document_id = uuid4()
    object_store_url = storage.put(src, Path(str(document_id)))

    crud = DocumentCrud(session, current_user.project_.id)
    document = Document(
        id=document_id,
        fname=src.filename,
        object_store_url=str(object_store_url),
    )
    source_document = crud.update(document)

    job_info: TransformationJobInfo | None = schedule_transformation(
        session=session,
        project_id=current_user.project_.id,
        source_format=source_format,
        target_format=target_format,
        actual_transformer=actual_transformer,
        source_document_id=source_document.id,
        callback_url=callback_url,
    )

    document_schema = DocumentPublic.model_validate(
        source_document, from_attributes=True
    )
    document_schema.signed_url = storage.get_signed_url(
        source_document.object_store_url
    )

    response = DocumentUploadResponse(
        **document_schema.model_dump(),
        transformation_job=job_info,
    )
    return APIResponse.success_response(response)


@router.delete(
    "/{doc_id}",
    description=load_description("documents/delete.md"),
    response_model=APIResponse[Message],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def remove_doc(
    session: SessionDep,
    current_user: AuthContextDep,
    doc_id: UUID = FastPath(description="Document to delete"),
):
    client = get_openai_client(
        session, current_user.organization_.id, current_user.project_.id
    )

    a_crud = OpenAIAssistantCrud(client)
    v_crud = OpenAIVectorStoreCrud(client)
    d_crud = DocumentCrud(session, current_user.project_.id)
    c_crud = CollectionCrud(session, current_user.project_.id)
    document = d_crud.read_one(doc_id)

    remote = pick_service_for_documennt(
        session, doc_id, a_crud, v_crud
    )  # assistant crud or vector store crud
    c_crud.delete(document, remote)
    d_crud.delete(doc_id)

    return APIResponse.success_response(
        Message(message="Document Deleted Successfully")
    )


@router.delete(
    "/{doc_id}/permanent",
    description=load_description("documents/permanent_delete.md"),
    response_model=APIResponse[Message],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def permanent_delete_doc(
    session: SessionDep,
    current_user: AuthContextDep,
    doc_id: UUID = FastPath(description="Document to permanently delete"),
):
    client = get_openai_client(
        session, current_user.organization_.id, current_user.project_.id
    )
    a_crud = OpenAIAssistantCrud(client)
    v_crud = OpenAIVectorStoreCrud(client)
    d_crud = DocumentCrud(session, current_user.project_.id)
    c_crud = CollectionCrud(session, current_user.project_.id)
    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)

    document = d_crud.read_one(doc_id)

    remote = pick_service_for_documennt(
        session, doc_id, a_crud, v_crud
    )  # assistant crud or vector store crud
    c_crud.delete(document, remote)

    storage.delete(document.object_store_url)
    d_crud.delete(doc_id)

    return APIResponse.success_response(
        Message(message="Document permanently deleted successfully")
    )


@router.get(
    "/{doc_id}",
    description=load_description("documents/info.md"),
    response_model=APIResponse[Union[DocumentPublic, TransformedDocumentPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def doc_info(
    session: SessionDep,
    current_user: AuthContextDep,
    doc_id: UUID = FastPath(description="Document to retrieve"),
    include_url: bool = Query(
        False, description="Include a signed URL to access the document"
    ),
):
    crud = DocumentCrud(session, current_user.project_.id)
    document = crud.read_one(doc_id)

    storage = (
        get_cloud_storage(session=session, project_id=current_user.project_.id)
        if include_url
        else None
    )

    doc_schema = build_document_schema(
        document=document,
        include_url=include_url,
        storage=storage,
    )

    return APIResponse.success_response(doc_schema)
