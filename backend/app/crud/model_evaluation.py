import logging
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import Float, cast
from sqlmodel import Session, select

from app.crud import fetch_by_id
from app.models import (
    ModelEvaluation,
    ModelEvaluationStatus,
    ModelEvaluationBase,
    ModelEvaluationUpdate,
)
from app.core.util import now


logger = logging.getLogger(__name__)


def create_model_evaluation(
    session: Session,
    request: ModelEvaluationBase,
    project_id: int,
    organization_id: int,
    status: ModelEvaluationStatus = ModelEvaluationStatus.pending,
) -> ModelEvaluation:
    fine_tuning_job = fetch_by_id(session, request.fine_tuning_id, project_id)

    if fine_tuning_job.fine_tuned_model and fine_tuning_job.test_data_s3_object is None:
        logger.warning(
            f"[create_model_evaluation] No fine tuned model or test data found for the given fine tuning ID | fine_tuning_id={request.fine_tuning_id}, project_id={project_id}"
        )
        raise HTTPException(404, "Fine tuned model not found")

    base_data = {
        "fine_tuning_id": request.fine_tuning_id,
        "system_prompt": fine_tuning_job.system_prompt,
        "base_model": fine_tuning_job.base_model,
        "split_ratio": fine_tuning_job.split_ratio,
        "fine_tuned_model": fine_tuning_job.fine_tuned_model,
        "document_id": fine_tuning_job.document_id,
        "test_data_s3_object": fine_tuning_job.test_data_s3_object,
        "project_id": project_id,
        "organization_id": organization_id,
        "status": status,
    }

    model_eval = ModelEvaluation(**base_data)
    model_eval.updated_at = now()

    session.add(model_eval)
    session.commit()
    session.refresh(model_eval)

    logger.info(
        f"[Create_fine_tuning_job]Created new model evaluation from fine tuning job ID={fine_tuning_job.id}, project_id={project_id}"
    )
    return model_eval


def fetch_by_eval_id(
    session: Session, eval_id: int, project_id: int
) -> ModelEvaluation:
    model_eval = session.exec(
        select(ModelEvaluation).where(
            ModelEvaluation.id == eval_id, ModelEvaluation.project_id == project_id
        )
    ).one_or_none()

    if model_eval is None:
        raise HTTPException(status_code=404, detail="Model evaluation not found")

    logger.info(
        f"[fetch_by_id]Fetched model evaluation for eval ID={model_eval.id}, project_id={project_id}"
    )
    return model_eval


def fetch_eval_by_doc_id(
    session: Session,
    document_id: UUID,
    project_id: int,
) -> list[ModelEvaluation]:
    query = (
        select(ModelEvaluation)
        .where(
            ModelEvaluation.document_id == document_id,
            ModelEvaluation.project_id == project_id,
        )
        .order_by(ModelEvaluation.updated_at.desc())
    )

    model_evals = session.exec(query).all()

    if not model_evals:
        raise HTTPException(status_code=404, detail="Model evaluation not found")

    logger.info(
        f"[fetch_eval_by_doc_id]Found {len(model_evals)} model evaluation(s) for document_id={document_id}, "
        f"project_id={project_id}"
    )

    return model_evals


def fetch_top_model_by_doc_id(
    session: Session, document_id: UUID, project_id: int
) -> ModelEvaluation:
    mcc_expr = cast(ModelEvaluation.score["mcc_score"].astext, Float)

    stmt = (
        select(ModelEvaluation)
        .where(
            ModelEvaluation.document_id == document_id,
            ModelEvaluation.project_id == project_id,
            ModelEvaluation.deleted_at.is_(None),
            ModelEvaluation.score.is_not(None),
            mcc_expr.is_not(None),
        )
        .order_by(mcc_expr.desc())
        .limit(1)
    )

    top_model = session.exec(stmt).first()

    if not top_model:
        logger.warning(
            f"[fetch_top_model_by_doc_id]No model evaluation found with populated score for document_id={document_id}, project_id={project_id}"
        )
        raise HTTPException(status_code=404, detail="No top model found")

    logger.info(
        f"[fetch_top_model_by_doc_id]Found top model evaluation for document_id={document_id}, "
        f"project_id={project_id}, sorted by MCC"
    )

    return top_model


def fetch_active_model_evals(
    session: Session,
    fine_tuning_id: int,
    project_id: int,
) -> list["ModelEvaluation"]:
    """
    Return all ACTIVE model evaluations for the given document & project.
    Active = status != failed AND not soft-deleted.
    """
    stmt = (
        select(ModelEvaluation)
        .where(
            ModelEvaluation.fine_tuning_id == fine_tuning_id,
            ModelEvaluation.project_id == project_id,
            ModelEvaluation.deleted_at.is_(None),
            ModelEvaluation.status != "failed",
        )
        .order_by(ModelEvaluation.inserted_at.desc())
    )

    return session.exec(stmt).all()


def update_model_eval(
    session: Session, eval_id: int, project_id: int, update: ModelEvaluationUpdate
) -> ModelEvaluation:
    model_eval = fetch_by_eval_id(session, eval_id, project_id)

    logger.info(
        f"[update_model_eval] Updating model evaluation ID={model_eval.id} with status={update.status}"
    )

    for key, value in update.model_dump(exclude_unset=True).items():
        setattr(model_eval, key, value)

    model_eval.updated_at = now()

    session.add(model_eval)
    session.commit()
    session.refresh(model_eval)

    logger.info(
        f"[update_model_eval] Successfully updated model evaluation ID={model_eval.id}"
    )
    return model_eval
