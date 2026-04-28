"""Assessment API routes."""

import asyncio
import io
import logging
import zipfile
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_feature, require_permission
from app.assessment.crud import (
    get_assessment_by_id,
    get_assessment_run_by_id,
    list_assessment_runs,
    list_assessments,
)
from app.assessment.dataset import upload_dataset as upload_assessment_dataset
from app.assessment.events import assessment_event_broker
from app.assessment.models import (
    AssessmentCreate,
    AssessmentDatasetResponse,
    AssessmentExportRow,
    AssessmentPublic,
    AssessmentResponse,
    AssessmentRun,
    AssessmentRunPublic,
)
from app.assessment.service import (
    retry_assessment,
    retry_assessment_run,
    start_assessment,
)
from app.assessment.utils import (
    build_export_response,
    build_json_export_rows,
    load_export_rows_for_run,
    serialize_export_rows,
    sort_export_rows,
)
from app.assessment.utils.export import _safe_filename_part
from app.assessment.validators import validate_dataset_file
from app.core.cloud import get_cloud_storage
from app.core.feature_flags import FeatureFlag
from app.core.storage_utils import generate_timestamped_filename
from app.crud.evaluations import get_dataset_by_id
from app.crud.evaluations import list_datasets as list_evaluation_datasets
from app.crud.evaluations.dataset import delete_dataset as delete_dataset_crud
from app.models.evaluation import EvaluationDataset
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/assessment",
    tags=["Assessment"],
    dependencies=[Depends(require_feature(FeatureFlag.ASSESSMENT))],
)


# ── Dataset routes ───────────────────────────────────────────────


def _dataset_to_response(
    dataset: EvaluationDataset,
    signed_url: str | None = None,
) -> AssessmentDatasetResponse:
    """Convert a dataset model to an AssessmentDatasetResponse."""
    metadata = dataset.dataset_metadata or {}
    return AssessmentDatasetResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        description=dataset.description,
        total_items=metadata.get("total_items_count", 0),
        file_extension=metadata.get("file_extension"),
        object_store_url=dataset.object_store_url,
        signed_url=signed_url,
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
    datasets = list_evaluation_datasets(
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
) -> APIResponse[AssessmentDatasetResponse]:
    """Get a specific assessment dataset."""
    dataset = get_dataset_by_id(
        session=session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=404, detail=f"Dataset {dataset_id} not found or not accessible"
        )

    signed_url = None
    if include_signed_url and dataset.object_store_url:
        storage = get_cloud_storage(
            session=session, project_id=auth_context.project_.id
        )
        signed_url = storage.get_signed_url(dataset.object_store_url)

    return APIResponse.success_response(
        data=_dataset_to_response(dataset, signed_url=signed_url)
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
    dataset = get_dataset_by_id(
        session=session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=404, detail=f"Dataset {dataset_id} not found or not accessible"
        )

    dataset_name = dataset.name
    error = delete_dataset_crud(session=session, dataset=dataset)
    if error:
        raise HTTPException(status_code=400, detail=error)

    return APIResponse.success_response(
        data={
            "message": f"Successfully deleted dataset '{dataset_name}' (id={dataset_id})",
            "dataset_id": dataset_id,
        }
    )


# ── Evaluation routes ────────────────────────────────────────────


@router.post(
    "/evaluations",
    description=load_description("assessment/create_evaluation.md"),
    response_model=APIResponse[AssessmentResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_evaluation(
    request: AssessmentCreate,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentResponse]:
    """Submit an assessment evaluation run."""
    logger.info(
        f"[create_evaluation] Assessment evaluation | "
        f"experiment={request.experiment_name} | "
        f"dataset_id={request.dataset_id} | "
        f"configs={len(request.configs)}"
    )

    result = start_assessment(
        session=session,
        request=request,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=result)


@router.post(
    "/assessments/{assessment_id}/retry",
    description=load_description("assessment/retry_assessment.md"),
    response_model=APIResponse[AssessmentResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def retry_assessment_manager(
    assessment_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentResponse]:
    """Retry a parent assessment using the same dataset/config inputs."""
    assessment = get_assessment_by_id(
        session=session,
        assessment_id=assessment_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    if not assessment:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment {assessment_id} not found or not accessible",
        )

    result = retry_assessment(
        session=session,
        assessment=assessment,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    return APIResponse.success_response(data=result)


@router.post(
    "/evaluations/{evaluation_id}/retry",
    description=load_description("assessment/retry_evaluation.md"),
    response_model=APIResponse[AssessmentResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def retry_assessment_evaluation(
    evaluation_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentResponse]:
    """Retry a single child assessment run using the same inputs."""
    run = get_assessment_run_by_id(
        session=session,
        run_id=evaluation_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment evaluation {evaluation_id} not found or not accessible",
        )

    result = retry_assessment_run(
        session=session,
        run=run,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    return APIResponse.success_response(data=result)


@router.get(
    "/events",
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    include_in_schema=False,
)
async def stream_assessment_events(
    _session: SessionDep,
    _auth_context: AuthContextDep,
) -> StreamingResponse:
    """SSE stream for assessment invalidation events."""

    async def event_stream():
        async for chunk in assessment_event_broker.subscribe():
            yield chunk
            await asyncio.sleep(0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/assessments",
    description=load_description("assessment/list_assessments.md"),
    response_model=APIResponse[list[AssessmentPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_assessment_managers(
    session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[AssessmentPublic]]:
    """List parent assessment manager rows."""
    assessments = list_assessments(
        session=session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        limit=limit,
        offset=offset,
    )

    return APIResponse.success_response(
        data=[
            AssessmentPublic.model_validate(assessment, from_attributes=True)
            for assessment in assessments
        ]
    )


@router.get(
    "/assessments/{assessment_id}",
    description=load_description("assessment/get_assessment.md"),
    response_model=APIResponse[AssessmentPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_assessment_manager(
    assessment_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentPublic]:
    """Get a specific parent assessment manager row."""
    assessment = get_assessment_by_id(
        session=session,
        assessment_id=assessment_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not assessment:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment {assessment_id} not found or not accessible",
        )

    return APIResponse.success_response(
        data=AssessmentPublic.model_validate(assessment, from_attributes=True)
    )


@router.get(
    "/assessments/{assessment_id}/results",
    description=load_description("assessment/export_assessment_results.md"),
    response_model=APIResponse[list[dict[str, Any]]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def export_assessment_results(
    assessment_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
    export_format: Literal["json", "csv", "xlsx"] = Query(default="json"),
) -> APIResponse[list[dict[str, Any]]] | StreamingResponse:
    """Return child-run results. For CSV/XLSX with multiple runs, returns a ZIP."""
    assessment = get_assessment_by_id(
        session=session,
        assessment_id=assessment_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    if not assessment:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment {assessment_id} not found or not accessible",
        )

    runs = list_assessment_runs(
        session=session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        assessment_id=assessment_id,
        limit=max(assessment.total_runs, 1),
        offset=0,
    )

    # Build per-run export data
    runs_with_rows: list[tuple[AssessmentRun, list[AssessmentExportRow]]] = []
    all_rows: list[AssessmentExportRow] = []
    for run in runs:
        rows = load_export_rows_for_run(session=session, run=run, assessment=assessment)
        if rows:
            runs_with_rows.append((run, sort_export_rows(rows)))
            all_rows.extend(rows)

    all_rows = sort_export_rows(all_rows)

    # JSON: return flat list
    if export_format == "json":
        return APIResponse.success_response(data=build_json_export_rows(all_rows))

    # Single run: return a single file directly
    if len(runs_with_rows) <= 1:
        return build_export_response(
            export_rows=all_rows,
            export_format=export_format,
            base_name=f"{assessment.experiment_name}_assessment_{assessment.id}_results",
        )

    # Multiple runs: ZIP with one file per run
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for run, rows in runs_with_rows:
            config_label = (
                f"config_v{run.config_version}"
                if run.config_version
                else f"run_{run.id}"
            )
            config_id_short = str(run.config_id)[:8] if run.config_id else ""
            file_base = _safe_filename_part(f"{config_label}_{config_id_short}")
            file_bytes, _ = serialize_export_rows(rows, export_format)
            zf.writestr(f"{file_base}.{export_format}", file_bytes)

    zip_buffer.seek(0)
    zip_filename = generate_timestamped_filename(
        _safe_filename_part(f"{assessment.experiment_name}_assessment_{assessment.id}"),
        extension="zip",
    )
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


@router.get(
    "/evaluations/{evaluation_id}/results",
    description=load_description("assessment/export_evaluation_results.md"),
    response_model=APIResponse[list[dict[str, Any]]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def export_assessment_run_results(
    evaluation_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
    export_format: Literal["json", "csv", "xlsx"] = Query(default="json"),
) -> APIResponse[list[dict[str, Any]]] | StreamingResponse:
    """Return flattened results for a single child assessment run."""
    run = get_assessment_run_by_id(
        session=session,
        run_id=evaluation_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment evaluation {evaluation_id} not found or not accessible",
        )

    assessment = None
    if run.assessment_id is not None:
        assessment = get_assessment_by_id(
            session=session,
            assessment_id=run.assessment_id,
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
        )

    export_rows = sort_export_rows(
        load_export_rows_for_run(
            session=session,
            run=run,
            assessment=assessment,
        )
    )

    if export_format != "json":
        return build_export_response(
            export_rows=export_rows,
            export_format=export_format,
            base_name=f"{run.run_name}_evaluation_{run.id}_results",
        )

    return APIResponse.success_response(data=build_json_export_rows(export_rows))


@router.get(
    "/evaluations",
    description=load_description("assessment/list_evaluations.md"),
    response_model=APIResponse[list[AssessmentRunPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_evaluations(
    session: SessionDep,
    auth_context: AuthContextDep,
    assessment_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> APIResponse[list[AssessmentRunPublic]]:
    """List assessment evaluation runs."""
    runs = list_assessment_runs(
        session=session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        assessment_id=assessment_id,
        limit=limit,
        offset=offset,
    )

    return APIResponse.success_response(
        data=[
            AssessmentRunPublic.model_validate(run, from_attributes=True)
            for run in runs
        ]
    )


@router.get(
    "/evaluations/{evaluation_id}",
    description=load_description("assessment/get_evaluation.md"),
    response_model=APIResponse[AssessmentRunPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_evaluation(
    evaluation_id: int,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentRunPublic]:
    """Get a specific assessment evaluation run."""
    run = get_assessment_run_by_id(
        session=session,
        run_id=evaluation_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment evaluation {evaluation_id} not found or not accessible",
        )

    return APIResponse.success_response(
        data=AssessmentRunPublic.model_validate(run, from_attributes=True)
    )
