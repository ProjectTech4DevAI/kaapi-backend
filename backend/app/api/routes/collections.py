import logging
from uuid import UUID
from typing import List

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from fastapi import Path as FastPath

from app.api.deps import SessionDep, AuthContextDep
from app.crud.collection.collection import CollectionNameConflictError
from app.api.permissions import Permission, require_permission
from app.core.telemetry import log_context
from app.core.rate_monitor import monitor_rate
from app.crud import (
    CollectionCrud,
    CollectionJobCrud,
    DocumentCollectionCrud,
)
from app.core.cloud import get_cloud_storage
from app.models import (
    CollectionJobStatus,
    CollectionActionType,
    CollectionJobCreate,
    CollectionJobPublic,
    CollectionJobImmediatePublic,
    CollectionWithDocsPublic,
)
from app.models.collection import (
    CreationRequest,
    CallbackRequest,
    DeletionRequest,
    CollectionPublic,
    CollectionUpdate,
)
from app.utils import APIResponse, load_description, validate_callback_url
from app.services.collections.helpers import ensure_unique_name, to_collection_public
from app.services.collections import (
    create_collection as create_service,
    delete_collection as delete_service,
)
from app.services.documents.helpers import build_document_schemas


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["Collections"])
collection_callback_router = APIRouter()


@collection_callback_router.post(
    "{$callback_url}",
    name="collection_callback",
)
def collection_callback_notification(body: APIResponse[CollectionJobPublic]):
    """
    Callback endpoint specification for collection creation/deletion.

    The callback will receive:
    - On success: APIResponse with success=True and data containing CollectionJobPublic
    - On failure: APIResponse with success=False and error message
    - metadata field will always be included if provided in the request
    """
    ...


@router.get(
    "",
    description=load_description("collections/list.md"),
    response_model=APIResponse[List[CollectionPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_collections(
    session: SessionDep,
    current_user: AuthContextDep,
) -> APIResponse[list[CollectionPublic]]:
    collection_crud = CollectionCrud(session, current_user.project_.id)
    rows = collection_crud.read_all()

    # Convert each collection to CollectionPublic with correct field mapping
    public_collections = [to_collection_public(collection) for collection in rows]

    return APIResponse[list[CollectionPublic]].success_response(public_collections)


@router.post(
    "",
    description=load_description("collections/create.md"),
    response_model=APIResponse[CollectionJobImmediatePublic],
    callbacks=collection_callback_router.routes,
    dependencies=[
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
        Depends(monitor_rate("collections")),
    ],
)
def create_collection(
    session: SessionDep,
    current_user: AuthContextDep,
    request: CreationRequest,
) -> APIResponse[CollectionJobImmediatePublic]:
    with log_context(
        tag="collection",
        system="collection",
        lifecycle="api.collection.create",
        action="create",
        project_id=current_user.project_.id,
        organization_id=current_user.organization_.id,
    ):
        if request.callback_url:
            validate_callback_url(str(request.callback_url))

        if request.name:
            ensure_unique_name(session, current_user.project_.id, request.name)

        unique_documents = list(dict.fromkeys(request.documents))

        collection_job_crud = CollectionJobCrud(session, current_user.project_.id)
        collection_job = collection_job_crud.create(
            CollectionJobCreate(
                action_type=CollectionActionType.CREATE,
                project_id=current_user.project_.id,
                status=CollectionJobStatus.PENDING,
                docs_num=len(unique_documents),
                documents=[str(doc_id) for doc_id in unique_documents],
            )
        )

        create_service.start_job(
            db=session,
            request=request,
            collection_job_id=collection_job.id,
            project_id=current_user.project_.id,
            organization_id=current_user.organization_.id,
        )

        return APIResponse[CollectionJobImmediatePublic].success_response(
            CollectionJobImmediatePublic.model_validate(collection_job),
        )


@router.delete(
    "/{collection_id}",
    description=load_description("collections/delete.md"),
    response_model=APIResponse[CollectionJobImmediatePublic],
    callbacks=collection_callback_router.routes,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def delete_collection(
    session: SessionDep,
    current_user: AuthContextDep,
    collection_id: UUID = FastPath(description="Collection to delete"),
    request: CallbackRequest | None = Body(default=None),
) -> APIResponse[CollectionJobImmediatePublic]:
    with log_context(
        tag="collection",
        system="collection",
        lifecycle="api.collection.delete",
        action="delete",
        collection_id=str(collection_id),
        project_id=current_user.project_.id,
        organization_id=current_user.organization_.id,
    ):
        if request and request.callback_url:
            validate_callback_url(str(request.callback_url))

        _ = CollectionCrud(session, current_user.project_.id).read_one_if_delete(
            collection_id
        )

        deletion_request = DeletionRequest(
            collection_id=collection_id,
            callback_url=request.callback_url if request else None,
        )

        collection_job_crud = CollectionJobCrud(session, current_user.project_.id)
        collection_job = collection_job_crud.create(
            CollectionJobCreate(
                action_type=CollectionActionType.DELETE,
                project_id=current_user.project_.id,
                status=CollectionJobStatus.PENDING,
                collection_id=collection_id,
            )
        )

        delete_service.start_job(
            db=session,
            request=deletion_request,
            collection_job_id=collection_job.id,
            project_id=current_user.project_.id,
            organization_id=current_user.organization_.id,
        )

        return APIResponse[CollectionJobImmediatePublic].success_response(
            CollectionJobImmediatePublic.model_validate(collection_job)
        )


@router.patch(
    "/{collection_id}",
    description=load_description("collections/update.md"),
    response_model=APIResponse[CollectionPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def update_collection(
    session: SessionDep,
    current_user: AuthContextDep,
    patch: CollectionUpdate,
    collection_id: UUID = FastPath(description="Collection to update"),
) -> APIResponse[CollectionPublic]:
    with log_context(
        tag="collection",
        system="collection",
        lifecycle="api.collection.update",
        action="update",
        collection_id=str(collection_id),
        project_id=current_user.project_.id,
        organization_id=current_user.organization_.id,
    ):
        collection_crud = CollectionCrud(session, current_user.project_.id)
        try:
            collection = collection_crud.update(collection_id, patch)
        except CollectionNameConflictError as err:
            raise HTTPException(
                status_code=409,
                detail=f"Collection '{err.name}' already exists. Choose a different name.",
            )

        logger.info(
            f"[update_collection] Collection updated | {{'collection_id': '{collection_id}'}}"
        )

        return APIResponse[CollectionPublic].success_response(
            to_collection_public(collection)
        )


@router.get(
    "/{collection_id}",
    description=load_description("collections/info.md"),
    response_model=APIResponse[CollectionWithDocsPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def collection_info(
    session: SessionDep,
    current_user: AuthContextDep,
    collection_id: UUID = FastPath(description="Collection to retrieve"),
    include_docs: bool = Query(
        True,
        description="If true, include documents linked to this collection",
    ),
    include_url: bool = Query(
        False, description="Include a signed URL to access the document"
    ),
    limit: int
    | None = Query(
        None,
        gt=0,
        le=500,
        description="Limit number of documents returned (default: all, max: 500)",
    ),
) -> APIResponse[CollectionWithDocsPublic]:
    collection_crud = CollectionCrud(session, current_user.project_.id)
    collection = collection_crud.read_one(collection_id)

    # Convert to CollectionPublic with correct field mapping, then to WithDocs
    collection_public = to_collection_public(collection)
    collection_with_docs = CollectionWithDocsPublic.model_validate(collection_public)

    if include_docs:
        document_collection_crud = DocumentCollectionCrud(session)
        documents = document_collection_crud.read(collection, skip=None, limit=limit)

        storage = None
        if include_url and documents:
            storage = get_cloud_storage(
                session=session, project_id=current_user.project_.id
            )

        collection_with_docs.documents = build_document_schemas(
            documents=documents, storage=storage, include_url=include_url
        )

    return APIResponse[CollectionWithDocsPublic].success_response(collection_with_docs)
