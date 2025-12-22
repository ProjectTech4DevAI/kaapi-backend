import csv
import io
import logging
import re
from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    Body,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from app.api.deps import AuthContextDep, SessionDep
from app.core.cloud import get_cloud_storage
from app.crud.config.version import ConfigVersionCrud
from app.crud.evaluations import (
    create_evaluation_dataset,
    create_evaluation_run,
    get_dataset_by_id,
    get_evaluation_run_by_id,
    list_datasets,
    start_evaluation_batch,
    upload_csv_to_object_store,
    upload_dataset_to_langfuse,
)
from app.crud.evaluations import list_evaluation_runs as list_evaluation_runs_crud
from app.crud.evaluations.core import save_score
from app.crud.evaluations.dataset import delete_dataset as delete_dataset_crud
from app.crud.evaluations.langfuse import fetch_trace_scores_from_langfuse
from app.models.evaluation import (
    DatasetUploadResponse,
    EvaluationRunPublic,
)
from app.models.llm.request import LLMCallConfig
from app.services.llm.jobs import resolve_config_blob
from app.services.llm.providers import LLMProvider
from app.utils import (
    APIResponse,
    get_langfuse_client,
    get_openai_client,
    load_description,
)


logger = logging.getLogger(__name__)

# File upload security constants
MAX_FILE_SIZE = 1024 * 1024  # 1 MB
ALLOWED_EXTENSIONS = {".csv"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/csv",
    "text/plain",  # Some systems report CSV as text/plain
}

router = APIRouter(tags=["Evaluation"])


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


def sanitize_dataset_name(name: str) -> str:
    """
    Sanitize dataset name for Langfuse compatibility.

    Langfuse has issues with spaces and special characters in dataset names.
    This function ensures the name can be both created and fetched.

    Rules:
    - Replace spaces with underscores
    - Replace hyphens with underscores
    - Keep only alphanumeric characters and underscores
    - Convert to lowercase for consistency
    - Remove leading/trailing underscores
    - Collapse multiple consecutive underscores into one

    Args:
        name: Original dataset name

    Returns:
        Sanitized dataset name safe for Langfuse

    Examples:
        "testing 0001" -> "testing_0001"
        "My Dataset!" -> "my_dataset"
        "Test--Data__Set" -> "test_data_set"
    """
    # Convert to lowercase
    sanitized = name.lower()

    # Replace spaces and hyphens with underscores
    sanitized = sanitized.replace(" ", "_").replace("-", "_")

    # Keep only alphanumeric characters and underscores
    sanitized = re.sub(r"[^a-z0-9_]", "", sanitized)

    # Collapse multiple underscores into one
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Ensure name is not empty
    if not sanitized:
        raise ValueError("Dataset name cannot be empty after sanitization")

    return sanitized


@router.post(
    "/evaluations/datasets",
    description=load_description("evaluation/upload_dataset.md"),
    response_model=APIResponse[DatasetUploadResponse],
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
    # Sanitize dataset name for Langfuse compatibility
    original_name = dataset_name
    try:
        dataset_name = sanitize_dataset_name(dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid dataset name: {str(e)}")

    if original_name != dataset_name:
        logger.info(
            f"[upload_dataset] Dataset name sanitized | '{original_name}' -> '{dataset_name}'"
        )

    logger.info(
        f"[upload_dataset] Uploading dataset | dataset={dataset_name} | "
        f"duplication_factor={duplication_factor} | org_id={auth_context.organization.id} | "
        f"project_id={auth_context.project.id}"
    )

    # Security validation: Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type. Only CSV files are allowed. Got: {file_ext}",
        )

    # Security validation: Check MIME type
    content_type = file.content_type
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type. Expected CSV, got: {content_type}",
        )

    # Security validation: Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB",
        )

    if file_size == 0:
        raise HTTPException(status_code=422, detail="Empty file uploaded")

    # Read CSV content
    csv_content = await file.read()

    # Step 1: Parse and validate CSV
    try:
        csv_text = csv_content.decode("utf-8")
        csv_reader = csv.DictReader(io.StringIO(csv_text))

        if not csv_reader.fieldnames:
            raise HTTPException(status_code=422, detail="CSV file has no headers")

        # Normalize headers for case-insensitive matching
        clean_headers = {
            field.strip().lower(): field for field in csv_reader.fieldnames
        }

        # Validate required headers (case-insensitive)
        if "question" not in clean_headers or "answer" not in clean_headers:
            raise HTTPException(
                status_code=422,
                detail=f"CSV must contain 'question' and 'answer' columns "
                f"Found columns: {csv_reader.fieldnames}",
            )

        # Get the actual column names from the CSV
        question_col = clean_headers["question"]
        answer_col = clean_headers["answer"]

        # Count original items
        original_items = []
        for row in csv_reader:
            question = row.get(question_col, "").strip()
            answer = row.get(answer_col, "").strip()
            if question and answer:
                original_items.append({"question": question, "answer": answer})

        if not original_items:
            raise HTTPException(
                status_code=422, detail="No valid items found in CSV file"
            )

        original_items_count = len(original_items)
        total_items_count = original_items_count * duplication_factor

        logger.info(
            f"[upload_dataset] Parsed items from CSV | original={original_items_count} | "
            f"total_with_duplication={total_items_count}"
        )

    except Exception as e:
        logger.error(f"[upload_dataset] Failed to parse CSV | {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Invalid CSV file: {e}")

    # Step 2: Upload to object store (if credentials configured)
    object_store_url = None
    try:
        storage = get_cloud_storage(
            session=_session, project_id=auth_context.project.id
        )
        object_store_url = upload_csv_to_object_store(
            storage=storage, csv_content=csv_content, dataset_name=dataset_name
        )
        if object_store_url:
            logger.info(
                f"[upload_dataset] Successfully uploaded CSV to object store | {object_store_url}"
            )
        else:
            logger.info(
                "[upload_dataset] Object store upload returned None | continuing without object store storage"
            )
    except Exception as e:
        logger.warning(
            f"[upload_dataset] Failed to upload CSV to object store (continuing without object store) | {e}",
            exc_info=True,
        )
        object_store_url = None

    # Step 3: Upload to Langfuse
    langfuse_dataset_id = None
    try:
        # Get Langfuse client
        langfuse = get_langfuse_client(
            session=_session,
            org_id=auth_context.organization.id,
            project_id=auth_context.project.id,
        )

        # Upload to Langfuse
        langfuse_dataset_id, _ = upload_dataset_to_langfuse(
            langfuse=langfuse,
            items=original_items,
            dataset_name=dataset_name,
            duplication_factor=duplication_factor,
        )

        logger.info(
            f"[upload_dataset] Successfully uploaded dataset to Langfuse | "
            f"dataset={dataset_name} | id={langfuse_dataset_id}"
        )

    except Exception as e:
        logger.error(
            f"[upload_dataset] Failed to upload dataset to Langfuse | {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to upload dataset to Langfuse: {e}"
        )

    # Step 4: Store metadata in database
    metadata = {
        "original_items_count": original_items_count,
        "total_items_count": total_items_count,
        "duplication_factor": duplication_factor,
    }

    dataset = create_evaluation_dataset(
        session=_session,
        name=dataset_name,
        description=description,
        dataset_metadata=metadata,
        object_store_url=object_store_url,
        langfuse_dataset_id=langfuse_dataset_id,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
    )

    logger.info(
        f"[upload_dataset] Successfully created dataset record in database | "
        f"id={dataset.id} | name={dataset_name}"
    )

    # Return response
    return APIResponse.success_response(
        data=DatasetUploadResponse(
            dataset_id=dataset.id,
            dataset_name=dataset_name,
            total_items=total_items_count,
            original_items=original_items_count,
            duplication_factor=duplication_factor,
            langfuse_dataset_id=langfuse_dataset_id,
            object_store_url=object_store_url,
        )
    )


@router.get(
    "/evaluations/datasets",
    description=load_description("evaluation/list_datasets.md"),
    response_model=APIResponse[list[DatasetUploadResponse]],
)
def list_datasets_endpoint(
    _session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = 50,
    offset: int = 0,
) -> APIResponse[list[DatasetUploadResponse]]:
    # Enforce maximum limit
    if limit > 100:
        limit = 100

    datasets = list_datasets(
        session=_session,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
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
)
def get_dataset(
    dataset_id: int,
    _session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[DatasetUploadResponse]:
    logger.info(
        f"[get_dataset] Fetching dataset | id={dataset_id} | "
        f"org_id={auth_context.organization.id} | "
        f"project_id={auth_context.project.id}"
    )

    dataset = get_dataset_by_id(
        session=_session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
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
)
def delete_dataset(
    dataset_id: int,
    _session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[dict]:
    logger.info(
        f"[delete_dataset] Deleting dataset | id={dataset_id} | "
        f"org_id={auth_context.organization.id} | "
        f"project_id={auth_context.project.id}"
    )

    success, message = delete_dataset_crud(
        session=_session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
    )

    if not success:
        # Check if it's a not found error or other error type
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
)
def evaluate(
    _session: SessionDep,
    auth_context: AuthContextDep,
    dataset_id: int = Body(..., description="ID of the evaluation dataset"),
    experiment_name: str = Body(
        ..., description="Name for this evaluation experiment/run"
    ),
    config_id: UUID = Body(..., description="Stored config ID"),
    config_version: int = Body(..., ge=1, description="Stored config version"),
) -> APIResponse[EvaluationRunPublic]:
    # Step 1: Fetch dataset from database
    dataset = get_dataset_by_id(
        session=_session,
        dataset_id=dataset_id,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found or not accessible to this "
            f"organization/project",
        )

    dataset_name = dataset.name

    # Get API clients
    openai_client = get_openai_client(
        session=_session,
        org_id=auth_context.organization.id,
        project_id=auth_context.project.id,
    )
    langfuse = get_langfuse_client(
        session=_session,
        org_id=auth_context.organization.id,
        project_id=auth_context.project.id,
    )

    # Validate dataset has Langfuse ID (should have been set during dataset creation)
    if not dataset.langfuse_dataset_id:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {dataset_id} does not have a Langfuse dataset ID. "
            "Please ensure Langfuse credentials were configured when the dataset was created.",
        )

    config_version_crud = ConfigVersionCrud(
        session=_session, config_id=config_id, project_id=auth_context.project.id
    )

    config, error = resolve_config_blob(
        config_crud=config_version_crud,
        config=LLMCallConfig(id=config_id, version=config_version),
    )
    if error:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to resolve config from stored config: {error}",
        )
    elif config.completion.provider != LLMProvider.OPENAI:
        raise HTTPException(
            status_code=422,
            detail="Only 'openai' provider is supported for evaluation configs",
        )

    logger.info("[evaluate] Successfully resolved config from config management")

    # Create EvaluationRun record with config references
    eval_run = create_evaluation_run(
        session=_session,
        run_name=experiment_name,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        config_id=config_id,
        config_version=config_version,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
    )

    # Start the batch evaluation
    try:
        eval_run = start_evaluation_batch(
            langfuse=langfuse,
            openai_client=openai_client,
            session=_session,
            eval_run=eval_run,
            config=config.completion.params,
        )

        logger.info(
            f"[evaluate] Evaluation started successfully | "
            f"batch_job_id={eval_run.batch_job_id} | total_items={eval_run.total_items}"
        )

        return APIResponse.success_response(data=eval_run)

    except Exception as e:
        logger.error(
            f"[evaluate] Failed to start evaluation | run_id={eval_run.id} | {e}",
            exc_info=True,
        )
        # Error is already handled in start_evaluation_batch
        _session.refresh(eval_run)
        return APIResponse.success_response(data=eval_run)


@router.get(
    "/evaluations",
    description=load_description("evaluation/list_evaluations.md"),
    response_model=APIResponse[list[EvaluationRunPublic]],
)
def list_evaluation_runs(
    _session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = 50,
    offset: int = 0,
) -> APIResponse[list[EvaluationRunPublic]]:
    logger.info(
        f"[list_evaluation_runs] Listing evaluation runs | "
        f"org_id={auth_context.organization.id} | "
        f"project_id={auth_context.project.id} | limit={limit} | offset={offset}"
    )

    return APIResponse.success_response(
        data=list_evaluation_runs_crud(
            session=_session,
            organization_id=auth_context.organization.id,
            project_id=auth_context.project.id,
            limit=limit,
            offset=offset,
        )
    )


@router.get(
    "/evaluations/{evaluation_id}",
    description=load_description("evaluation/get_evaluation.md"),
    response_model=APIResponse[EvaluationRunPublic],
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
    logger.info(
        f"[get_evaluation_run_status] Fetching status for evaluation run | "
        f"evaluation_id={evaluation_id} | "
        f"org_id={auth_context.organization.id} | "
        f"project_id={auth_context.project.id} | "
        f"get_trace_info={get_trace_info} | "
        f"resync_score={resync_score}"
    )

    if resync_score and not get_trace_info:
        raise HTTPException(
            status_code=400,
            detail="resync_score=true requires get_trace_info=true",
        )

    eval_run = get_evaluation_run_by_id(
        session=_session,
        evaluation_id=evaluation_id,
        organization_id=auth_context.organization.id,
        project_id=auth_context.project.id,
    )

    if not eval_run:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Evaluation run {evaluation_id} not found or not accessible "
                "to this organization"
            ),
        )

    if get_trace_info:
        # Only fetch trace info for completed evaluations
        if eval_run.status != "completed":
            return APIResponse.failure_response(
                error=f"Trace info is only available for completed evaluations. "
                f"Current status: {eval_run.status}",
                data=eval_run,
            )

        # Check if we already have cached scores (before any slow operations)
        has_cached_score = eval_run.score is not None and "traces" in eval_run.score
        if not resync_score and has_cached_score:
            return APIResponse.success_response(data=eval_run)

        # Get Langfuse client (needs session for credentials lookup)
        langfuse = get_langfuse_client(
            session=_session,
            org_id=auth_context.organization.id,
            project_id=auth_context.project.id,
        )

        # Capture data needed for Langfuse fetch and DB update
        dataset_name = eval_run.dataset_name
        run_name = eval_run.run_name
        eval_run_id = eval_run.id
        org_id = auth_context.organization.id
        project_id = auth_context.project.id

        # Session is no longer needed - slow Langfuse API calls happen here
        # without holding the DB connection
        try:
            score = fetch_trace_scores_from_langfuse(
                langfuse=langfuse,
                dataset_name=dataset_name,
                run_name=run_name,
            )
        except ValueError as e:
            # Run not found in Langfuse - return eval_run with error
            logger.warning(
                f"[get_evaluation_run_status] Run not found in Langfuse | "
                f"evaluation_id={evaluation_id} | error={e}"
            )
            return APIResponse.failure_response(error=str(e), data=eval_run)
        except Exception as e:
            logger.error(
                f"[get_evaluation_run_status] Failed to fetch trace info | "
                f"evaluation_id={evaluation_id} | error={e}",
                exc_info=True,
            )
            return APIResponse.failure_response(
                error=f"Failed to fetch trace info from Langfuse: {str(e)}",
                data=eval_run,
            )

        # Open new session just for the score commit
        eval_run = save_score(
            eval_run_id=eval_run_id,
            organization_id=org_id,
            project_id=project_id,
            score=score,
        )

        if not eval_run:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation run {evaluation_id} not found after score update",
            )

    return APIResponse.success_response(data=eval_run)
