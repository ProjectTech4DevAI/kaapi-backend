from uuid import UUID

import pytest
from sqlmodel import Session
from fastapi import HTTPException

from app.tests.utils.utils import get_project, get_non_existent_id
from app.tests.utils.test_data import (
    create_test_model_evaluation,
    create_test_finetuning_job_with_extra_fields,
)
from app.models import (
    ModelEvaluation,
    ModelEvaluationBase,
    ModelEvaluationUpdate,
    ModelEvaluationStatus,
)
from app.crud import (
    create_model_evaluation,
    fetch_by_eval_id,
    fetch_eval_by_doc_id,
    fetch_top_model_by_doc_id,
    fetch_active_model_evals,
    update_model_eval,
)


def test_create_model_evaluation(db: Session) -> None:
    project = get_project(db, "Dalgo")

    fine_tune_jobs, _ = create_test_finetuning_job_with_extra_fields(db, [0.5])
    fine_tune = fine_tune_jobs[0]

    job_request = ModelEvaluationBase(
        fine_tuning_id=fine_tune.id,
        system_prompt=fine_tune.system_prompt,
        base_model=fine_tune.base_model,
        fine_tuned_model=fine_tune.fine_tuned_model,
        document_id=fine_tune.document_id,
        test_data_s3_object=fine_tune.test_data_s3_object,
        status="pending",
    )

    created_eval = create_model_evaluation(
        session=db,
        request=job_request,
        project_id=project.id,
        organization_id=project.organization_id,
    )

    assert created_eval.id is not None
    assert created_eval.status == "pending"
    assert created_eval.document_id == fine_tune.document_id
    assert created_eval.fine_tuned_model == fine_tune.fine_tuned_model
    assert created_eval.test_data_s3_object == fine_tune.test_data_s3_object


def test_fetch_by_eval_id_success(db: Session) -> None:
    model_evals = create_test_model_evaluation(db)
    model_eval = model_evals[0]
    result = fetch_by_eval_id(
        db, eval_id=model_eval.id, project_id=model_eval.project_id
    )
    assert result.id == model_eval.id


def test_fetch_by_eval_id_not_found(db: Session) -> None:
    with pytest.raises(HTTPException) as exc:
        fetch_by_eval_id(
            db, eval_id=get_non_existent_id(db, ModelEvaluation), project_id=1
        )
    assert exc.value.status_code == 404


def test_fetch_eval_by_doc_id_success(db: Session) -> None:
    model_evals = create_test_model_evaluation(db)
    doc_id = model_evals[0].document_id

    result = fetch_eval_by_doc_id(
        db, document_id=doc_id, project_id=model_evals[0].project_id
    )
    assert len(result) > 0


def test_fetch_eval_by_doc_id_not_found(db: Session) -> None:
    valid_uuid = UUID("c5d479e2-66a5-40b8-aa76-4a2290b6d1f3")
    with pytest.raises(HTTPException) as exc:
        fetch_eval_by_doc_id(db, document_id=valid_uuid, project_id=1)
    assert exc.value.status_code == 404


def test_fetch_top_model_by_doc_id_success(db: Session) -> None:
    model_evals = create_test_model_evaluation(db)
    model_eval = model_evals[0]
    model_eval.score = {"mcc_score": 0.8}
    db.flush()

    doc_id = model_eval.document_id

    result = fetch_top_model_by_doc_id(
        db, document_id=doc_id, project_id=model_evals[0].project_id
    )
    assert result.id == model_eval.id


def test_fetch_top_model_by_doc_id_not_found(db: Session) -> None:
    valid_uuid = UUID("c5d479e2-66a5-40b8-aa76-4a2290b6d1f3")
    with pytest.raises(HTTPException) as exc:
        fetch_top_model_by_doc_id(db, document_id=valid_uuid, project_id=1)
    assert exc.value.status_code == 404


def test_fetch_active_model_evals(db: Session) -> None:
    model_evals = create_test_model_evaluation(db)
    active_evals = fetch_active_model_evals(
        db,
        fine_tuning_id=model_evals[0].fine_tuning_id,
        project_id=model_evals[0].project_id,
    )
    assert len(active_evals) > 0
    assert all(eval.status != "failed" for eval in active_evals)


def test_update_model_eval_success(db: Session) -> None:
    model_evals = create_test_model_evaluation(db)
    model_eval = model_evals[0]

    update = ModelEvaluationUpdate(status="completed")
    updated_eval = update_model_eval(
        db, eval_id=model_eval.id, project_id=model_eval.project_id, update=update
    )

    assert updated_eval.status == "completed"
    assert updated_eval.updated_at is not None


def test_update_model_eval_not_found(db: Session) -> None:
    project = get_project(db)

    with pytest.raises(HTTPException) as exc:
        update_model_eval(
            db,
            eval_id=get_non_existent_id(db, ModelEvaluation),
            project_id=project.id,
            update=ModelEvaluationUpdate(status=ModelEvaluationStatus.completed),
        )

    assert exc.value.status_code == 404
    assert exc.value.detail == "Model evaluation not found"
