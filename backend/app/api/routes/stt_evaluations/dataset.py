"""STT dataset API routes."""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.file import get_files_by_ids
from app.crud.language import get_language_by_id
from app.crud.stt_evaluations import (
    get_stt_dataset_by_id,
    list_stt_datasets,
    get_samples_by_dataset_id,
)
from app.models.stt_evaluation import (
    STTDatasetCreate,
    STTDatasetPublic,
    STTDatasetWithSamples,
    STTSamplePublic,
)
from app.services.stt_evaluations.dataset import upload_stt_dataset
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/datasets",
    response_model=APIResponse[STTDatasetPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Create STT dataset",
    description=load_description("stt_evaluation/create_dataset.md"),
)
def create_dataset(
    _session: SessionDep,
    auth_context: AuthContextDep,
    dataset_create: STTDatasetCreate = Body(...),
) -> APIResponse[STTDatasetPublic]:
    """Create an STT evaluation dataset."""
    # Validate language_id if provided
    if dataset_create.language_id is not None:
        language = get_language_by_id(
            session=_session, language_id=dataset_create.language_id
        )
        if not language:
            raise HTTPException(
                status_code=400, detail="Invalid language_id: language not found"
            )

    dataset, samples = upload_stt_dataset(
        session=_session,
        name=dataset_create.name,
        samples=dataset_create.samples,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        description=dataset_create.description,
        language_id=dataset_create.language_id,
    )

    return APIResponse.success_response(
        data=STTDatasetPublic(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            type=dataset.type,
            language_id=dataset.language_id,
            object_store_url=dataset.object_store_url,
            dataset_metadata=dataset.dataset_metadata,
            sample_count=len(samples),
            organization_id=dataset.organization_id,
            project_id=dataset.project_id,
            inserted_at=dataset.inserted_at,
            updated_at=dataset.updated_at,
        )
    )


@router.get(
    "/datasets",
    response_model=APIResponse[list[STTDatasetPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="List STT datasets",
    description=load_description("stt_evaluation/list_datasets.md"),
)
def list_datasets(
    _session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> APIResponse[list[STTDatasetPublic]]:
    """List STT evaluation datasets."""
    datasets, total = list_stt_datasets(
        session=_session,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        limit=limit,
        offset=offset,
    )

    return APIResponse.success_response(
        data=datasets,
        metadata={"total": total, "limit": limit, "offset": offset},
    )


@router.get(
    "/datasets/{dataset_id}",
    response_model=APIResponse[STTDatasetWithSamples],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Get STT dataset",
    description=load_description("stt_evaluation/get_dataset.md"),
)
def get_dataset(
    _session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int,
    include_samples: bool = Query(True, description="Include samples in response"),
    sample_limit: int = Query(100, ge=1, le=1000, description="Max samples to return"),
    sample_offset: int = Query(0, ge=0, description="Sample offset"),
) -> APIResponse[STTDatasetWithSamples]:
    """Get an STT evaluation dataset."""
    dataset = get_stt_dataset_by_id(
        session=_session,
        dataset_id=dataset_id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    samples = []
    samples_total = (dataset.dataset_metadata or {}).get("sample_count", 0)

    if include_samples:
        sample_records = get_samples_by_dataset_id(
            session=_session,
            dataset_id=dataset_id,
            org_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            limit=sample_limit,
            offset=sample_offset,
        )

        # Fetch file records to get object_store_url
        file_ids = [s.file_id for s in sample_records]
        file_records = get_files_by_ids(
            session=_session,
            file_ids=file_ids,
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
        )
        file_map = {f.id: f for f in file_records}

        samples = [
            STTSamplePublic(
                id=s.id,
                file_id=s.file_id,
                object_store_url=file_map.get(s.file_id).object_store_url
                if s.file_id in file_map
                else None,
                language_id=s.language_id,
                ground_truth=s.ground_truth,
                sample_metadata=s.sample_metadata,
                dataset_id=s.dataset_id,
                organization_id=s.organization_id,
                project_id=s.project_id,
                inserted_at=s.inserted_at,
                updated_at=s.updated_at,
            )
            for s in sample_records
        ]

    return APIResponse.success_response(
        data=STTDatasetWithSamples(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            type=dataset.type,
            language_id=dataset.language_id,
            object_store_url=dataset.object_store_url,
            dataset_metadata=dataset.dataset_metadata,
            organization_id=dataset.organization_id,
            project_id=dataset.project_id,
            inserted_at=dataset.inserted_at,
            updated_at=dataset.updated_at,
            samples=samples,
        ),
        metadata={"samples_total": samples_total},
    )
