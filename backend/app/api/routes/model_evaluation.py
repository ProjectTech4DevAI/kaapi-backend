import logging
import time
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlmodel import Session

from app.crud import (
    fetch_by_id,
    create_model_evaluation,
    fetch_active_model_evals,
    fetch_eval_by_doc_id,
    update_model_eval,
    fetch_top_model_by_doc_id,
)
from app.models import (
    ModelEvaluationBase,
    ModelEvaluationCreate,
    ModelEvaluationStatus,
    ModelEvaluationUpdate,
    ModelEvaluationPublic,
)
from app.core.db import engine
from app.core.cloud import get_cloud_storage
from app.core.finetune.evaluation import ModelEvaluator
from app.utils import get_openai_client, APIResponse, load_description
from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model_evaluation", tags=["Model Evaluation"])


def attach_prediction_file_url(model_obj, storage):
    """
    Given a model-like object and a storage client,
    attach a signed prediction data file URL (if available).
    """
    s3_key = getattr(model_obj, "prediction_data_s3_object", None)
    prediction_data_file_url = storage.get_signed_url(s3_key) if s3_key else None

    return model_obj.model_copy(
        update={"prediction_data_file_url": prediction_data_file_url}
    )


def run_model_evaluation(
    eval_id: int,
    current_user: AuthContextDep,
):
    start_time = time.time()
    logger.info(
        f"[run_model_evaluation] Starting | eval_id={eval_id}, project_id={current_user.project_.id}"
    )

    with Session(engine) as db:
        client = get_openai_client(
            db, current_user.organization_.id, current_user.project_.id
        )
        storage = get_cloud_storage(session=db, project_id=current_user.project_.id)

        try:
            model_eval = update_model_eval(
                session=db,
                eval_id=eval_id,
                project_id=current_user.project_.id,
                update=ModelEvaluationUpdate(status=ModelEvaluationStatus.running),
            )

            evaluator = ModelEvaluator(
                fine_tuned_model=model_eval.fine_tuned_model,
                test_data_s3_object=model_eval.test_data_s3_object,
                system_prompt=model_eval.system_prompt,
                client=client,
                storage=storage,
            )
            result = evaluator.run()

            update_model_eval(
                session=db,
                eval_id=eval_id,
                project_id=current_user.project_.id,
                update=ModelEvaluationUpdate(
                    score=result["evaluation_score"],
                    prediction_data_s3_object=result["prediction_data_s3_object"],
                    status=ModelEvaluationStatus.completed,
                ),
            )

            elapsed = time.time() - start_time
            logger.info(
                f"[run_model_evaluation] Completed | eval_id={eval_id}, project_id={current_user.project_.id}, elapsed={elapsed:.2f}s"
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"[run_model_evaluation] Failed | eval_id={eval_id}, project_id={current_user.project_.id}: {e}"
            )
            db.rollback()
            update_model_eval(
                session=db,
                eval_id=eval_id,
                project_id=current_user.project_.id,
                update=ModelEvaluationUpdate(
                    status=ModelEvaluationStatus.failed,
                    error_message="failed during background job processing:"
                    + error_msg,
                ),
            )


@router.post(
    "/evaluate_models/",
    response_model=APIResponse,
    description=load_description("model_evaluation/evaluate.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def evaluate_models(
    request: ModelEvaluationCreate,
    background_tasks: BackgroundTasks,
    session: SessionDep,
    current_user: AuthContextDep,
):
    """
    Start evaluations for one or more fine-tuning jobs.

    Request:{ fine_tuning_ids: list[int] } (one or many).

    Process:
        For each ID, it fetches the fine-tuned model and its testing file from fine tuning table,
        then queues a background task that runs predictions on the test set
        and computes evaluation scores.

    Response:
        APIResponse with the created/active evaluation records and a success message.
    """
    client = get_openai_client(
        session, current_user.organization_.id, current_user.project_.id
    )  # keeping this here for checking if the user's validated OpenAI key is present or not,
    # even though the client will be initialized separately inside the background task

    if not request.fine_tuning_ids:
        logger.error(
            f"[evaluate_model] No fine tuning IDs provided | project_id:{current_user.project_.id}"
        )
        raise HTTPException(status_code=400, detail="No fine-tuned job IDs provided")

    evaluations: list[ModelEvaluationPublic] = []

    for job_id in request.fine_tuning_ids:
        fine_tuning_job = fetch_by_id(session, job_id, current_user.project_.id)
        active_evaluations = fetch_active_model_evals(
            session, job_id, current_user.project_.id
        )

        if active_evaluations:
            logger.info(
                f"[evaluate_model] Skipping creation for {job_id}. Active evaluation exists, project_id:{current_user.project_.id}"
            )
            evaluations.extend(
                ModelEvaluationPublic.model_validate(ev) for ev in active_evaluations
            )
            continue

        model_eval = create_model_evaluation(
            session=session,
            request=ModelEvaluationBase(fine_tuning_id=fine_tuning_job.id),
            project_id=current_user.project_.id,
            organization_id=current_user.organization_.id,
            status=ModelEvaluationStatus.pending,
        )

        evaluations.append(ModelEvaluationPublic.model_validate(model_eval))

        logger.info(
            f"[evaluate_model] Created evaluation for fine_tuning_id {job_id} with eval ID={model_eval.id}, project_id:{current_user.project_.id}"
        )

        background_tasks.add_task(run_model_evaluation, model_eval.id, current_user)

    response_data = [
        {
            "id": ev.id,
            "fine_tuning_id": ev.fine_tuning_id,
            "fine_tuned_model": getattr(ev, "fine_tuned_model", None),
            "document_id": getattr(ev, "document_id", None),
            "status": ev.status,
        }
        for ev in evaluations
    ]

    return APIResponse.success_response(
        {"message": "Model evaluation(s) started successfully", "data": response_data}
    )


@router.get(
    "/{document_id}/top_model",
    response_model=APIResponse[ModelEvaluationPublic],
    response_model_exclude_none=True,
    description=load_description("model_evaluation/get_top_model.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_top_model_by_doc_id(
    document_id: UUID,
    session: SessionDep,
    current_user: AuthContextDep,
):
    """
    Return the top model trained on the given document_id, ranked by
    Matthews correlation coefficient (MCC) across all evaluations.
    """
    logger.info(
        f"[get_top_model_by_doc_id] Fetching top model for document_id={document_id}, "
        f"project_id={current_user.project_.id}"
    )

    top_model = fetch_top_model_by_doc_id(
        session, document_id, current_user.project_.id
    )
    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)

    top_model = attach_prediction_file_url(top_model, storage)

    return APIResponse.success_response(top_model)


@router.get(
    "/{document_id}",
    response_model=APIResponse[list[ModelEvaluationPublic]],
    response_model_exclude_none=True,
    description=load_description("model_evaluation/list_by_document.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_evaluations_by_doc_id(
    document_id: UUID,
    session: SessionDep,
    current_user: AuthContextDep,
):
    """
    Return all model evaluations for the given document_id within the current project.
    """
    logger.info(
        f"[get_evaluations_by_doc_id] Fetching evaluations for document_id={document_id}, "
        f"project_id={current_user.project_.id}"
    )

    evaluations = fetch_eval_by_doc_id(session, document_id, current_user.project_.id)
    storage = get_cloud_storage(session=session, project_id=current_user.project_.id)

    updated_evaluations = [
        attach_prediction_file_url(ev, storage) for ev in evaluations
    ]

    return APIResponse.success_response(updated_evaluations)
