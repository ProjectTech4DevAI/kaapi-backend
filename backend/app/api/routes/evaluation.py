"""Evaluation API routes."""

import logging

from fastapi import (
    APIRouter,
    Body,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    Depends,
)

from app.api.deps import AuthContextDep, SessionDep
from app.crud.evaluations import (
    get_dataset_by_id,
    list_datasets,
)
from app.crud.evaluations import list_evaluation_runs as list_evaluation_runs_crud
from app.crud.evaluations.dataset import delete_dataset as delete_dataset_crud
from app.models.evaluation import (
    DatasetUploadResponse,
    EvaluationRunPublic,
)
from app.services.evaluation import (
    get_evaluation_with_scores,
    start_evaluation,
    upload_dataset,
    validate_csv_file,
)
from app.utils import (
    APIResponse,
    load_description,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluation"])


def _dataset_to_response(dataset) -> DatasetUploadResponse:
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
    "/evaluations/datasets",
    description=load_description("evaluation/upload_dataset.md"),
    response_model=APIResponse[DatasetUploadResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def upload_dataset_endpoint(
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
    dataset = upload_dataset(
        session=_session,
        csv_content=csv_content,
        dataset_name=dataset_name,
        description=description,
        duplication_factor=duplication_factor,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
    )

    return APIResponse.success_response(data=_dataset_to_response(dataset))


@router.get(
    "/evaluations/datasets",
    description=load_description("evaluation/list_datasets.md"),
    response_model=APIResponse[list[DatasetUploadResponse]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_datasets_endpoint(
    _session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = 50,
    offset: int = 0,
) -> APIResponse[list[DatasetUploadResponse]]:
    """List evaluation datasets."""
    # Enforce maximum limit
    if limit > 100:
        limit = 100

    datasets = list_datasets(
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
    "/evaluations/datasets/{dataset_id}",
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
    "/evaluations/datasets/{dataset_id}",
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

    success, message = delete_dataset_crud(
        session=_session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not success:
        if "not found" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        else:
            raise HTTPException(status_code=400, detail=message)

    logger.info(f"[delete_dataset] Successfully deleted dataset | id={dataset_id}")
    return APIResponse.success_response(
        data={"message": message, "dataset_id": dataset_id}
    )


@router.post(
    "/evaluations",
    description=load_description("evaluation/create_evaluation.md"),
    response_model=APIResponse[EvaluationRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def evaluate(
    _session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int = Body(..., description="ID of the evaluation dataset"),
    experiment_name: str = Body(
        ..., description="Name for this evaluation experiment/run"
    ),
    config: dict = Body(default_factory=dict, description="Evaluation configuration"),
    assistant_id: str
    | None = Body(
        None, description="Optional assistant ID to fetch configuration from"
    ),
) -> APIResponse[EvaluationRunPublic]:
    """Start an evaluation run."""
    eval_run = start_evaluation(
        session=_session,
        dataset_id=dataset_id,
        experiment_name=experiment_name,
        config=config,
        assistant_id=assistant_id,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
    )

    return APIResponse.success_response(data=eval_run)


@router.get(
    "/evaluations",
    description=load_description("evaluation/list_evaluations.md"),
    response_model=APIResponse[list[EvaluationRunPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_evaluation_runs(
    _session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = 50,
    offset: int = 0,
) -> APIResponse[list[EvaluationRunPublic]]:
    """List evaluation runs."""
    logger.info(
        f"[list_evaluation_runs] Listing evaluation runs | "
        f"org_id={auth_context.organization_.id} | "
        f"project_id={auth_context.project_.id} | limit={limit} | offset={offset}"
    )

    return APIResponse.success_response(
        data=list_evaluation_runs_crud(
            session=_session,
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            limit=limit,
            offset=offset,
        )
    )


@router.get(
    "/evaluations/{evaluation_id}",
    description=load_description("evaluation/get_evaluation.md"),
    response_model=APIResponse[EvaluationRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_evaluation_run_status(
    evaluation_id: int,
    _session: SessionDep,
    auth_context: AuthContextDep,
    get_trace_info: bool = Query(
        False,
        description=(
            "If true, fetch and include Langfuse trace scores with Q&A context. "
            "On first request, data is fetched from Langfuse and cached. "
            "Subsequent requests return cached data."
        ),
    ),
    resync_score: bool = Query(
        False,
        description=(
            "If true, clear cached scores and re-fetch from Langfuse. "
            "Useful when new evaluators have been added or scores have been updated. "
            "Requires get_trace_info=true."
        ),
    ),
) -> APIResponse[EvaluationRunPublic]:
    """Get evaluation run status with optional trace info."""
    if resync_score and not get_trace_info:
        raise HTTPException(
            status_code=400,
            detail="resync_score=true requires get_trace_info=true",
        )

    eval_run, error = get_evaluation_with_scores(
        session=_session,
        evaluation_id=evaluation_id,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
        get_trace_info=get_trace_info,
        resync_score=resync_score,
    )

    if not eval_run:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Evaluation run {evaluation_id} not found or not accessible "
                "to this organization"
            ),
        )

    if error:
        return APIResponse.failure_response(error=error, data=eval_run)
    return APIResponse.success_response(data=eval_run)
