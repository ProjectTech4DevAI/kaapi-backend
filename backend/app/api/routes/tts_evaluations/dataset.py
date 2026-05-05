"""TTS dataset API routes."""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.cloud import get_cloud_storage
from app.crud.language import get_language_by_id
from app.crud.tts_evaluations import (
    get_tts_dataset_by_id,
    list_tts_datasets,
)
from app.models.tts_evaluation import (
    TTSDatasetCreate,
    TTSDatasetPublic,
)
from app.services.tts_evaluations.dataset import upload_tts_dataset
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/datasets",
    response_model=APIResponse[TTSDatasetPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Create TTS dataset",
    description=load_description("tts_evaluation/create_dataset.md"),
)
def create_dataset(
    session: SessionDep,
    auth_context: AuthContextDep,
    dataset_create: TTSDatasetCreate = Body(...),
) -> APIResponse[TTSDatasetPublic]:
    """Create a TTS evaluation dataset."""
    # Validate language_id if provided
    if dataset_create.language_id is not None:
        get_language_by_id(session=session, language_id=dataset_create.language_id)

    dataset = upload_tts_dataset(
        session=session,
        name=dataset_create.name,
        samples=dataset_create.samples,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        description=dataset_create.description,
        language_id=dataset_create.language_id,
    )

    return APIResponse.success_response(data=TTSDatasetPublic.from_model(dataset))


@router.get(
    "/datasets",
    response_model=APIResponse[list[TTSDatasetPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="List TTS datasets",
    description=load_description("tts_evaluation/list_datasets.md"),
)
def list_datasets(
    session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    include_signed_url: bool = Query(
        False, description="Include signed URL for dataset files"
    ),
) -> APIResponse[list[TTSDatasetPublic]]:
    """List TTS evaluation datasets."""
    datasets, total = list_tts_datasets(
        session=session,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        limit=limit,
        offset=offset,
    )

    storage = None
    if include_signed_url:
        storage = get_cloud_storage(
            session=session, project_id=auth_context.project_.id
        )

    data = []
    for dataset in datasets:
        signed_url = None
        if storage and dataset.object_store_url:
            signed_url = storage.get_signed_url(dataset.object_store_url)
        data.append(TTSDatasetPublic.from_model(dataset, signed_url=signed_url))

    return APIResponse.success_response(
        data=data,
        metadata={"total": total, "limit": limit, "offset": offset},
    )


@router.get(
    "/datasets/{dataset_id}",
    response_model=APIResponse[TTSDatasetPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Get TTS dataset",
    description=load_description("tts_evaluation/get_dataset.md"),
)
def get_dataset(
    session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int,
    include_signed_url: bool = Query(
        False, description="Include signed URL for dataset file"
    ),
) -> APIResponse[TTSDatasetPublic]:
    """Get a TTS evaluation dataset."""
    dataset = get_tts_dataset_by_id(
        session=session,
        dataset_id=dataset_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    signed_url = None
    if include_signed_url and dataset.object_store_url:
        storage = get_cloud_storage(
            session=session, project_id=auth_context.project_.id
        )
        signed_url = storage.get_signed_url(dataset.object_store_url)

    return APIResponse.success_response(
        data=TTSDatasetPublic.from_model(dataset, signed_url=signed_url),
        metadata={
            "sample_count": (dataset.dataset_metadata or {}).get("sample_count", 0)
        },
    )
