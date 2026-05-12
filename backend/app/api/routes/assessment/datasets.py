"""Assessment dataset endpoints."""

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.cloud import get_cloud_storage
from app.crud.assessment.dataset import (
    delete_assessment_dataset,
    get_assessment_dataset_by_id,
    list_assessment_datasets,
)
from app.models.assessment import (
    AssessmentDatasetPreview,
    AssessmentDatasetResponse,
)
from app.models.evaluation import EvaluationDataset
from app.services.assessment.dataset import (
    preview_dataset as preview_assessment_dataset,
)
from app.services.assessment.dataset import upload_dataset as upload_assessment_dataset
from app.services.assessment.validators import validate_dataset_file
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


def _dataset_to_response(
    dataset: EvaluationDataset,
    signed_url: str | None = None,
    preview: AssessmentDatasetPreview | None = None,
) -> AssessmentDatasetResponse:
    metadata = dataset.dataset_metadata or {}
    return AssessmentDatasetResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        description=dataset.description,
        total_items=metadata.get("total_items_count", 0),
        file_extension=metadata.get("file_extension"),
        object_store_url=dataset.object_store_url,
        signed_url=signed_url,
        preview=preview,
    )


@router.post(
    "/datasets",
    description=load_description("assessment/upload_dataset.md"),
    response_model=APIResponse[AssessmentDatasetResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def upload_dataset(
    session: SessionDep,
    auth_context: AuthContextDep,
    file: UploadFile = File(
        ..., description="CSV or Excel file to upload as a dataset"
    ),
    dataset_name: str = Form(..., description="Name for the dataset"),
    description: str | None = Form(None, description="Optional dataset description"),
) -> APIResponse[AssessmentDatasetResponse]:
    """Upload an assessment dataset (any CSV/Excel file, no column requirements)."""
    file_content, file_ext = await validate_dataset_file(file)

    dataset = upload_assessment_dataset(
        session=session,
        file_content=file_content,
        file_ext=file_ext,
        dataset_name=dataset_name,
        description=description,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=_dataset_to_response(dataset))


@router.get(
    "/datasets",
    description=load_description("assessment/list_datasets.md"),
    response_model=APIResponse[list[AssessmentDatasetResponse]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_datasets(
    session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = Query(
        default=50, ge=1, le=100, description="Maximum number of datasets to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of datasets to skip"),
) -> APIResponse[list[AssessmentDatasetResponse]]:
    """List assessment datasets."""
    datasets = list_assessment_datasets(
        session=session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        limit=limit,
        offset=offset,
    )

    return APIResponse.success_response(
        data=[_dataset_to_response(dataset) for dataset in datasets]
    )


@router.get(
    "/datasets/{dataset_id}",
    description=load_description("assessment/get_dataset.md"),
    response_model=APIResponse[AssessmentDatasetResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_dataset(
    dataset_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
    include_signed_url: bool = Query(
        False, description="Include a signed URL for downloading the raw file from S3"
    ),
    limit_rows: int
    | None = Query(
        None,
        ge=1,
        le=100,
        description=(
            "If set, fetch the underlying file and include a preview of the first "
            "N data rows plus column headers. Skip to avoid the file download."
        ),
    ),
) -> APIResponse[AssessmentDatasetResponse]:
    """Get a specific assessment dataset."""
    dataset = get_assessment_dataset_by_id(
        session=session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    signed_url = None
    if include_signed_url and dataset.object_store_url:
        storage = get_cloud_storage(
            session=session, project_id=auth_context.project_.id
        )
        signed_url = storage.get_signed_url(dataset.object_store_url)

    preview: AssessmentDatasetPreview | None = None
    if limit_rows is not None:
        headers, rows = preview_assessment_dataset(
            session=session,
            dataset=dataset,
            project_id=auth_context.project_.id,
            limit=limit_rows,
        )
        preview = AssessmentDatasetPreview(
            headers=headers,
            rows=rows,
            returned_rows=len(rows),
            truncated=len(rows) >= limit_rows,
        )

    return APIResponse.success_response(
        data=_dataset_to_response(dataset, signed_url=signed_url, preview=preview)
    )


@router.delete(
    "/datasets/{dataset_id}",
    description=load_description("assessment/delete_dataset.md"),
    response_model=APIResponse[dict],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def delete_dataset(
    dataset_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[dict]:
    """Delete an assessment dataset."""
    dataset = get_assessment_dataset_by_id(
        session=session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    dataset_name = dataset.name
    error = delete_assessment_dataset(session=session, dataset=dataset)
    if error:
        raise HTTPException(status_code=400, detail=error)

    return APIResponse.success_response(
        data={
            "message": (
                f"Successfully deleted dataset '{dataset_name}' (id={dataset_id})"
            ),
            "dataset_id": dataset_id,
        }
    )
