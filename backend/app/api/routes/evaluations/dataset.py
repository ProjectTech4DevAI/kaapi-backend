"""Evaluation dataset API routes."""

import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.evaluations import (
    get_dataset_by_id,
    list_datasets as list_evaluation_datasets,
)
from app.crud.evaluations.dataset import delete_dataset as delete_dataset_crud
from app.models.evaluation import DatasetUploadResponse, EvaluationDataset
from app.services.evaluations import (
    upload_dataset as upload_evaluation_dataset,
    validate_csv_file,
)
from app.utils import (
    APIResponse,
    load_description,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _dataset_to_response(dataset: EvaluationDataset) -> DatasetUploadResponse:
    """Convert a dataset model to a DatasetUploadResponse."""
    return DatasetUploadResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        total_items=dataset.dataset_metadata.get("total_items_count", 0),
        original_items=dataset.dataset_metadata.get("original_items_count", 0),
        duplication_factor=dataset.dataset_metadata.get("duplication_factor", 1),
        langfuse_dataset_id=dataset.langfuse_dataset_id,
        object_store_url=dataset.object_store_url,
    )


@router.post(
    "/",
    description=load_description("evaluation/upload_dataset.md"),
    response_model=APIResponse[DatasetUploadResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def upload_dataset(
    _session: SessionDep,
    auth_context: AuthContextDep,
    file: UploadFile = File(
        ..., description="CSV file with 'question' and 'answer' columns"
    ),
    dataset_name: str = Form(..., description="Name for the dataset"),
    description: str | None = Form(None, description="Optional dataset description"),
    duplication_factor: int = Form(
        default=1,
        ge=1,
        le=5,
        description="Number of times to duplicate each item (min: 1, max: 5)",
    ),
) -> APIResponse[DatasetUploadResponse]:
    """Upload an evaluation dataset."""
    # Validate and read CSV file
    csv_content = await validate_csv_file(file)

    # Upload dataset using service
    dataset = upload_evaluation_dataset(
        session=_session,
        csv_content=csv_content,
        dataset_name=dataset_name,
        description=description,
        duplication_factor=duplication_factor,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=_dataset_to_response(dataset))


@router.get(
    "/",
    description=load_description("evaluation/list_datasets.md"),
    response_model=APIResponse[list[DatasetUploadResponse]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_datasets(
    _session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = Query(
        default=50, ge=1, le=100, description="Maximum number of datasets to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of datasets to skip"),
) -> APIResponse[list[DatasetUploadResponse]]:
    """List evaluation datasets."""
    datasets = list_evaluation_datasets(
        session=_session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        limit=limit,
        offset=offset,
    )

    return APIResponse.success_response(
        data=[_dataset_to_response(dataset) for dataset in datasets]
    )


@router.get(
    "/{dataset_id}",
    description=load_description("evaluation/get_dataset.md"),
    response_model=APIResponse[DatasetUploadResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_dataset(
    dataset_id: int,
    _session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[DatasetUploadResponse]:
    """Get a specific evaluation dataset."""
    logger.info(
        f"[get_dataset] Fetching dataset | id={dataset_id} | "
        f"org_id={auth_context.organization_.id} | "
        f"project_id={auth_context.project_.id}"
    )

    dataset = get_dataset_by_id(
        session=_session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=404, detail=f"Dataset {dataset_id} not found or not accessible"
        )

    return APIResponse.success_response(data=_dataset_to_response(dataset))


@router.delete(
    "/{dataset_id}",
    description=load_description("evaluation/delete_dataset.md"),
    response_model=APIResponse[dict],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def delete_dataset(
    dataset_id: int,
    _session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[dict]:
    """Delete an evaluation dataset."""
    logger.info(
        f"[delete_dataset] Deleting dataset | id={dataset_id} | "
        f"org_id={auth_context.organization_.id} | "
        f"project_id={auth_context.project_.id}"
    )

    dataset = get_dataset_by_id(
        session=_session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=404, detail=f"Dataset {dataset_id} not found or not accessible"
        )
    dataset_name = dataset.name
    error = delete_dataset_crud(session=_session, dataset=dataset)

    if error:
        raise HTTPException(status_code=400, detail=error)

    logger.info(f"[delete_dataset] Successfully deleted dataset | id={dataset_id}")
    return APIResponse.success_response(
        data={
            "message": f"Successfully deleted dataset '{dataset_name}' (id={dataset_id})",
            "dataset_id": dataset_id,
        }
    )
