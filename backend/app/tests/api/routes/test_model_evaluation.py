from typing import Any
from unittest.mock import patch

from sqlmodel import Session
from fastapi.testclient import TestClient

from app.tests.utils.test_data import (
    create_test_finetuning_job_with_extra_fields,
    create_test_model_evaluation,
)


@patch("app.api.routes.model_evaluation.run_model_evaluation")
def test_evaluate_model(
    mock_run_eval: Any,
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
) -> None:
    fine_tuned, _ = create_test_finetuning_job_with_extra_fields(db, [0.5])
    body = {"fine_tuning_ids": [fine_tuned[0].id]}

    resp = client.post(
        "/api/v1/model_evaluation/evaluate_models/",
        json=body,
        headers=user_api_key_header,
    )
    assert resp.status_code == 200, resp.text

    j = resp.json()
    evals = j["data"]["data"]
    assert len(evals) == 1
    assert evals[0]["status"] == "pending"

    mock_run_eval.assert_called_once()
    assert mock_run_eval.call_args[0][0] == evals[0]["id"]


def test_evaluate_model_finetuning_not_found(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    invalid_fine_tune_id = 9999

    body = {"fine_tuning_ids": [invalid_fine_tune_id]}

    response = client.post(
        "/api/v1/model_evaluation/evaluate_models/",
        json=body,
        headers=user_api_key_header,
    )

    assert response.status_code == 404
    json_data = response.json()
    assert json_data["error"] == f"Job not found"


def test_top_model_by_doc(
    client: TestClient, db: Session, user_api_key_header: dict[str, str]
) -> None:
    model_evals = create_test_model_evaluation(db)
    model_eval = model_evals[0]

    model_eval.score = {
        "mcc_score": 0.85,
    }
    db.flush()

    response = client.get(
        f"/api/v1/model_evaluation/{model_eval.document_id}/top_model",
        headers=user_api_key_header,
    )

    assert response.status_code == 200
    json_data = response.json()

    assert json_data["data"]["score"] == {
        "mcc_score": 0.85,
    }
    assert json_data["data"]["fine_tuned_model"] == model_eval.fine_tuned_model
    assert json_data["data"]["document_id"] == str(model_eval.document_id)

    assert json_data["data"]["id"] == model_eval.id


def test_get_top_model_by_doc_id_no_score(
    client: TestClient, db: Session, user_api_key_header: dict[str, str]
) -> None:
    model_evals = create_test_model_evaluation(db)

    document_id = model_evals[0].document_id

    response = client.get(
        f"/api/v1/model_evaluation/{document_id}/top_model", headers=user_api_key_header
    )

    assert response.status_code == 404

    json_data = response.json()
    assert json_data["error"] == "No top model found"


def test_get_evals_by_doc_id(
    client: TestClient, db: Session, user_api_key_header: dict[str, str]
) -> None:
    model_evals = create_test_model_evaluation(db)
    document_id = model_evals[0].document_id

    response = client.get(
        f"/api/v1/model_evaluation/{document_id}", headers=user_api_key_header
    )

    assert response.status_code == 200
    json_data = response.json()

    assert json_data["success"] is True
    assert json_data["data"] is not None
    assert len(json_data["data"]) == 2

    evaluations = json_data["data"]
    assert all(eval["document_id"] == str(document_id) for eval in evaluations)
    assert all(eval["status"] == "pending" for eval in evaluations)
