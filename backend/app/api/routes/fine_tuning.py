from typing import Optional
import logging
import time
from uuid import UUID, uuid4
from pathlib import Path

import openai
from sqlmodel import Session
from fastapi import (
    APIRouter,
    HTTPException,
    BackgroundTasks,
    File,
    Form,
    UploadFile,
    Depends,
)

from app.models import (
    FineTuningJobCreate,
    FineTuningJobPublic,
    FineTuningUpdate,
    FineTuningStatus,
    Document,
    ModelEvaluationBase,
    ModelEvaluationStatus,
)
from app.core.cloud import get_cloud_storage
from app.crud.document.document import DocumentCrud
from app.utils import (
    get_openai_client,
    handle_openai_error,
    mask_string,
    load_description,
    APIResponse,
)
from app.crud import (
    create_fine_tuning_job,
    fetch_by_id,
    update_finetune_job,
    fetch_by_document_id,
    create_model_evaluation,
    fetch_active_model_evals,
)
from app.core.db import engine
from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.finetune.preprocessing import DataPreprocessor
from app.api.routes.model_evaluation import run_model_evaluation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fine_tuning", tags=["Fine Tuning"])


OPENAI_TO_INTERNAL_STATUS = {
    "validating_files": FineTuningStatus.running,
    "queued": FineTuningStatus.running,
    "running": FineTuningStatus.running,
    "succeeded": FineTuningStatus.completed,
    "failed": FineTuningStatus.failed,
    "cancelled": FineTuningStatus.cancelled,
}


def process_fine_tuning_job(
    job_id: int,
    ratio: float,
    current_user: AuthContextDep,
    request: FineTuningJobCreate,
):
    start_time = time.time()
    project_id = current_user.project_.id
    fine_tune = None

    logger.info(
        f"[process_fine_tuning_job]Starting fine-tuning job processing | job_id={job_id}, project_id={project_id}|"
    )
    with Session(engine) as session:
        try:
            fine_tune = fetch_by_id(session, job_id, project_id)

            client = get_openai_client(
                session, current_user.organization_.id, project_id
            )
            storage = get_cloud_storage(
                session=session, project_id=current_user.project_.id
            )
            document_crud = DocumentCrud(session, current_user.project_.id)
            document = document_crud.read_one(request.document_id)
            preprocessor = DataPreprocessor(
                document, storage, ratio, request.system_prompt
            )
            result = preprocessor.process()
            train_data_temp_filepath = result["train_jsonl_temp_filepath"]
            train_data_s3_object = result["train_csv_s3_object"]
            test_data_s3_object = result["test_csv_s3_object"]

            try:
                with open(train_data_temp_filepath, "rb") as train_f:
                    uploaded_train = client.files.create(
                        file=train_f, purpose="fine-tune"
                    )
                logger.info(
                    f"[process_fine_tuning_job] File uploaded to OpenAI successfully | "
                    f"job_id={job_id}, project_id={project_id}|"
                )
            except openai.OpenAIError as e:
                error_msg = handle_openai_error(e)
                logger.error(
                    f"[process_fine_tuning_job] Failed to upload to OpenAI: {error_msg} | "
                    f"job_id={job_id}, project_id={project_id}|"
                )
                update_finetune_job(
                    session=session,
                    job=fine_tune,
                    update=FineTuningUpdate(
                        status=FineTuningStatus.failed,
                        error_message="Error while uploading file to openai : "
                        + error_msg,
                    ),
                )
                return
            finally:
                preprocessor.cleanup()

            training_file_id = uploaded_train.id

            try:
                job = client.fine_tuning.jobs.create(
                    training_file=training_file_id, model=request.base_model
                )
                logger.info(
                    f"[process_fine_tuning_job] OpenAI fine-tuning job created | "
                    f"provider_job_id={mask_string(job.id)}, job_id={job_id}, project_id={project_id}|"
                )
            except openai.OpenAIError as e:
                error_msg = handle_openai_error(e)
                logger.error(
                    f"[process_fine_tuning_job] Failed to create OpenAI fine-tuning job: {error_msg} | "
                    f"job_id={job_id}, project_id={project_id}|"
                )
                update_finetune_job(
                    session=session,
                    job=fine_tune,
                    update=FineTuningUpdate(
                        status=FineTuningStatus.failed,
                        error_message="Error while creating an openai fine tuning job : "
                        + error_msg,
                    ),
                )
                return

            update_finetune_job(
                session=session,
                job=fine_tune,
                update=FineTuningUpdate(
                    training_file_id=training_file_id,
                    train_data_s3_object=train_data_s3_object,
                    test_data_s3_object=test_data_s3_object,
                    split_ratio=ratio,
                    provider_job_id=job.id,
                    status=FineTuningStatus.running,
                ),
            )

            end_time = time.time()
            duration = end_time - start_time

            logger.info(
                f"[process_fine_tuning_job] Fine-tuning job processed successfully | "
                f"time_taken={duration:.2f}s, job_id={job_id}, project_id={project_id}|"
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"[process_fine_tuning_job] Background job failure: {e} | "
                f"job_id={job_id}, project_id={project_id}|"
            )
            update_finetune_job(
                session=session,
                job=fine_tune,
                update=FineTuningUpdate(
                    status=FineTuningStatus.failed,
                    error_message="Error while processing the background job : "
                    + error_msg,
                ),
            )


@router.post(
    "/fine_tune",
    description=load_description("fine_tuning/create.md"),
    response_model=APIResponse,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def fine_tune_from_CSV(
    session: SessionDep,
    current_user: AuthContextDep,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file to use for fine-tuning"),
    base_model: str = Form(...),
    split_ratio: str = Form(...),
    system_prompt: str = Form(...),
):
    # Parse split ratios
    try:
        split_ratios = [float(r.strip()) for r in split_ratio.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid split_ratio format: {e}")

    # Validate file is CSV
    if not file.filename.lower().endswith(".csv") and file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    get_openai_client(  # Used here only to validate the user's OpenAI key;
        # the actual client is re-initialized separately inside the background task
        session,
        current_user.organization_.id,
        current_user.project_.id,
    )

    # Upload the file to storage and create document
    # ToDo: create a helper function and then use it rather than doing things in router
    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)
    document_id = uuid4()
    object_store_url = storage.put(file, Path(str(document_id)))

    # Create document in database
    document_crud = DocumentCrud(session, current_user.project_.id)
    document = Document(
        id=document_id,
        fname=file.filename,
        object_store_url=str(object_store_url),
    )
    created_document = document_crud.update(document)

    # Create FineTuningJobCreate request object
    request = FineTuningJobCreate(
        document_id=created_document.id,
        base_model=base_model,
        split_ratio=split_ratios,
        system_prompt=system_prompt.strip(),
    )

    results = []

    for ratio in split_ratios:
        job, created = create_fine_tuning_job(
            session=session,
            request=request,
            split_ratio=ratio,
            organization_id=current_user.organization_.id,
            project_id=current_user.project_.id,
        )
        results.append((job, created))

        if created:
            background_tasks.add_task(
                process_fine_tuning_job, job.id, ratio, current_user, request
            )

    if not results:
        logger.error(
            f"[fine_tune_from_CSV]All fine-tuning job creations failed for document_id={request.document_id}, project_id={current_user.project_.id}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to create or fetch any fine-tuning jobs."
        )

    job_infos = [
        {
            "id": job.id,
            "document_id": job.document_id,
            "split_ratio": job.split_ratio,
            "status": job.status,
        }
        for job, _ in results
    ]

    created_count = sum(c for _, c in results)
    total = len(results)
    message = (
        "Fine-tuning job(s) started."
        if created_count == total
        else "Active fine-tuning job(s) already exists."
        if created_count == 0
        else f"Started {created_count} job(s); {total - created_count} active fine-tuning job(s) already exists."
    )

    return APIResponse.success_response({"message": message, "jobs": job_infos})


@router.get(
    "/{fine_tuning_id}/refresh",
    description=load_description("fine_tuning/retrieve.md"),
    response_model=APIResponse[FineTuningJobPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def refresh_fine_tune_status(
    fine_tuning_id: int,
    background_tasks: BackgroundTasks,
    session: SessionDep,
    current_user: AuthContextDep,
):
    project_id = current_user.project_.id
    job = fetch_by_id(session, fine_tuning_id, project_id)
    client = get_openai_client(session, current_user.organization_.id, project_id)
    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)

    if job.provider_job_id is not None:
        try:
            openai_job = client.fine_tuning.jobs.retrieve(job.provider_job_id)
        except openai.OpenAIError as e:
            error_msg = handle_openai_error(e)
            logger.error(
                f"[Retrieve_fine_tune_status] Failed to retrieve OpenAI job | "
                f"provider_job_id={mask_string(job.provider_job_id)}, "
                f"error={error_msg}, fine_tuning_id={fine_tuning_id}, project_id={project_id}"
            )
            raise HTTPException(
                status_code=502, detail=f"OpenAI API error: {error_msg}"
            )

        mapped_status: Optional[str] = OPENAI_TO_INTERNAL_STATUS.get(
            getattr(openai_job, "status", None)
        )

        openai_error = getattr(openai_job, "error", None)
        openai_error_msg = (
            getattr(openai_error, "message", None) if openai_error else None
        )

        update_payload = FineTuningUpdate(
            status=mapped_status or job.status,
            fine_tuned_model=getattr(openai_job, "fine_tuned_model", None),
            error_message=openai_error_msg,
        )

        # Check if status is changing from running to completed
        is_newly_completed = (
            job.status == FineTuningStatus.running
            and update_payload.status == FineTuningStatus.completed
        )

        if (
            job.status != update_payload.status
            or job.fine_tuned_model != update_payload.fine_tuned_model
            or job.error_message != update_payload.error_message
        ):
            job = update_finetune_job(session=session, job=job, update=update_payload)

        # If the job just completed, automatically trigger evaluation
        if is_newly_completed:
            logger.info(
                f"[refresh_fine_tune_status] Fine-tuning job completed, triggering evaluation | "
                f"fine_tuning_id={fine_tuning_id}, project_id={project_id}"
            )

            # Check if there's already an active evaluation for this job
            active_evaluations = fetch_active_model_evals(
                session, fine_tuning_id, project_id
            )

            if not active_evaluations:
                # Create a new evaluation
                model_eval = create_model_evaluation(
                    session=session,
                    request=ModelEvaluationBase(fine_tuning_id=fine_tuning_id),
                    project_id=project_id,
                    organization_id=current_user.organization_.id,
                    status=ModelEvaluationStatus.pending,
                )

                # Queue the evaluation task
                background_tasks.add_task(
                    run_model_evaluation, model_eval.id, current_user
                )

                logger.info(
                    f"[refresh_fine_tune_status] Created and queued evaluation | "
                    f"eval_id={model_eval.id}, fine_tuning_id={fine_tuning_id}, project_id={project_id}"
                )
            else:
                logger.info(
                    f"[refresh_fine_tune_status] Skipping evaluation creation - active evaluation exists | "
                    f"fine_tuning_id={fine_tuning_id}, project_id={project_id}"
                )

    job = job.model_copy(
        update={
            "train_data_file_url": storage.get_signed_url(job.train_data_s3_object)
            if job.train_data_s3_object
            else None,
            "test_data_file_url": storage.get_signed_url(job.test_data_s3_object)
            if job.test_data_s3_object
            else None,
        }
    )

    return APIResponse.success_response(job)


@router.get(
    "/{document_id}",
    description="Retrieves all fine-tuning jobs associated with the given document ID for the current project",
    response_model=APIResponse[list[FineTuningJobPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def retrieve_jobs_by_document(
    document_id: UUID, session: SessionDep, current_user: AuthContextDep
):
    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)
    project_id = current_user.project_.id
    jobs = fetch_by_document_id(session, document_id, project_id)
    if not jobs:
        logger.warning(
            f"[retrive_job_by_document]No fine-tuning jobs found for document_id={document_id}, project_id={project_id}"
        )
        raise HTTPException(
            status_code=404,
            detail="No fine-tuning jobs found for the given document ID",
        )
    updated_jobs = []
    for job in jobs:
        train_url = (
            storage.get_signed_url(job.train_data_s3_object)
            if job.train_data_s3_object
            else None
        )
        test_url = (
            storage.get_signed_url(job.test_data_s3_object)
            if job.test_data_s3_object
            else None
        )

        updated_job = job.model_copy(
            update={
                "train_data_file_url": train_url,
                "test_data_file_url": test_url,
            }
        )
        updated_jobs.append(updated_job)

    return APIResponse.success_response(updated_jobs)
