"""STT dataset API routes."""

import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.stt_evaluations import (
    create_stt_dataset,
    create_stt_samples,
    get_stt_dataset_by_id,
    list_stt_datasets,
    get_samples_by_dataset_id,
    get_sample_count_for_dataset,
    update_dataset_metadata,
)
from app.models.stt_evaluation import (
    STTDatasetCreate,
    STTDatasetPublic,
    STTDatasetWithSamples,
    STTSampleCreate,
    STTSamplePublic,
)
from app.utils import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/datasets",
    response_model=APIResponse[STTDatasetPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    summary="Create STT dataset",
    description="""
Create a new STT evaluation dataset with audio samples.

Each sample requires:
- **object_store_url**: S3 URL of the audio file (from /evaluations/stt/files/audio endpoint)
- **language**: ISO 639-1 language code (optional)
- **ground_truth**: Reference transcription (optional, for Phase 2 WER/CER)
""",
)
def create_dataset(
    _session: SessionDep,
    auth_context: AuthContextDep,
    dataset_create: STTDatasetCreate = Body(...),
) -> APIResponse[STTDatasetPublic]:
    """Create an STT evaluation dataset."""
    logger.info(
        f"[create_dataset] Creating STT dataset | "
        f"name: {dataset_create.name}, sample_count: {len(dataset_create.samples)}"
    )

    # Create dataset
    dataset = create_stt_dataset(
        session=_session,
        name=dataset_create.name,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        description=dataset_create.description,
        language=dataset_create.language,
        dataset_metadata={
            "sample_count": len(dataset_create.samples),
            "has_ground_truth_count": sum(
                1 for s in dataset_create.samples if s.ground_truth
            ),
        },
    )

    # Create samples
    samples = create_stt_samples(
        session=_session,
        dataset_id=dataset.id,
        org_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        samples=dataset_create.samples,
    )

    return APIResponse.success_response(
        data=STTDatasetPublic(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            type=dataset.type,
            language=dataset.language,
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
    description="List all STT evaluation datasets for the current project.",
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
    description="Get an STT dataset with its samples.",
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
    samples_total = 0

    if include_samples:
        sample_records, samples_total = get_samples_by_dataset_id(
            session=_session,
            dataset_id=dataset_id,
            org_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            limit=sample_limit,
            offset=sample_offset,
        )

        samples = [
            STTSamplePublic(
                id=s.id,
                object_store_url=s.object_store_url,
                language=s.language,
                ground_truth=s.ground_truth,
                duration_seconds=s.duration_seconds,
                sample_metadata=s.sample_metadata,
                dataset_id=s.dataset_id,
                organization_id=s.organization_id,
                project_id=s.project_id,
                inserted_at=s.inserted_at,
                updated_at=s.updated_at,
            )
            for s in sample_records
        ]
    else:
        samples_total = get_sample_count_for_dataset(
            session=_session, dataset_id=dataset_id
        )

    return APIResponse.success_response(
        data=STTDatasetWithSamples(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            type=dataset.type,
            language=dataset.language,
            dataset_metadata=dataset.dataset_metadata,
            sample_count=samples_total,
            organization_id=dataset.organization_id,
            project_id=dataset.project_id,
            inserted_at=dataset.inserted_at,
            updated_at=dataset.updated_at,
            samples=samples,
        ),
        metadata={"samples_total": samples_total},
    )
