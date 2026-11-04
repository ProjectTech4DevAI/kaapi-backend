from uuid import UUID, uuid4
from unittest.mock import patch
from typing import Any

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext
import pytest
from app.models import Collection, CollectionJob, CollectionJobStatus, Document
from app.models.collection import CreationRequest


def _create_test_document(
    db: Session, project_id: int, file_size: float = 1
) -> Document:
    doc = Document(
        id=uuid4(),
        fname="test_document.txt",
        object_store_url="s3://test-bucket/test_document.txt",
        project_id=project_id,
        file_size_kb=file_size,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@patch("app.api.routes.collections.create_service.start_job")
def test_collection_creation_calls_start_job_and_returns_job(
    mock_start_job: Any,
    client: TestClient,
    user_api_key_header: dict[str, str],
    user_api_key: TestAuthContext,
    db: Session,
) -> None:
    doc = _create_test_document(db, user_api_key.project_id, file_size=2)

    creation_data = CreationRequest(
        documents=[doc.id],
        callback_url=None,
    )

    resp = client.post(
        f"{settings.API_V1_STR}/collections",
        json=creation_data.model_dump(mode="json"),
        headers=user_api_key_header,
    )

    assert resp.status_code == 200
    body = resp.json()

    data = body["data"]
    assert data["status"] == CollectionJobStatus.PENDING
    assert data["job_inserted_at"]
    assert data["job_updated_at"]

    mock_start_job.assert_called_once()
    kwargs = mock_start_job.call_args.kwargs
    assert "db" in kwargs
    assert kwargs["project_id"] == user_api_key.project_id
    assert kwargs["organization_id"] == user_api_key.organization_id
    assert "with_assistant" not in kwargs

    returned_job_id = UUID(data["job_id"])
    assert kwargs["collection_job_id"] == returned_job_id

    assert kwargs["request"].model_dump(mode="json") == creation_data.model_dump(
        mode="json"
    )


@patch("app.api.routes.collections.create_service.start_job")
def test_duplicate_in_flight_name_returns_existing_job(
    mock_start_job: Any,
    client: TestClient,
    user_api_key_header: dict[str, str],
    user_api_key: TestAuthContext,
    db: Session,
) -> None:
    doc = _create_test_document(db, user_api_key.project_id)
    payload = {"name": "kb-retry", "documents": [str(doc.id)]}
    url = f"{settings.API_V1_STR}/collections"

    first = client.post(url, json=payload, headers=user_api_key_header)
    second = client.post(url, json=payload, headers=user_api_key_header)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["data"]["job_id"] == first.json()["data"]["job_id"]
    mock_start_job.assert_called_once()


@patch("app.api.routes.collections.create_service.start_job")
def test_existing_collection_name_still_returns_conflict(
    mock_start_job: Any,
    client: TestClient,
    user_api_key_header: dict[str, str],
    user_api_key: TestAuthContext,
    db: Session,
) -> None:
    doc = _create_test_document(db, user_api_key.project_id)
    db.add(
        Collection(
            provider="openai",
            knowledge_base_id="vs_existing",
            knowledge_base_provider="OpenAI",
            name="kb-taken",
            project_id=user_api_key.project_id,
        )
    )
    db.commit()

    resp = client.post(
        f"{settings.API_V1_STR}/collections",
        json={"name": "kb-taken", "documents": [str(doc.id)]},
        headers=user_api_key_header,
    )

    assert resp.status_code == 409
    mock_start_job.assert_not_called()


@patch("app.api.routes.collections.create_service.start_job")
def test_enqueue_failure_marks_job_failed_and_frees_name(
    mock_start_job: Any,
    client: TestClient,
    user_api_key_header: dict[str, str],
    user_api_key: TestAuthContext,
    db: Session,
) -> None:
    doc = _create_test_document(db, user_api_key.project_id)
    payload = {"name": "kb-enqueue", "documents": [str(doc.id)]}
    url = f"{settings.API_V1_STR}/collections"

    mock_start_job.side_effect = RuntimeError("broker down")
    with pytest.raises(RuntimeError):
        client.post(url, json=payload, headers=user_api_key_header)

    failed = db.exec(
        select(CollectionJob).where(CollectionJob.name == "kb-enqueue")
    ).one()
    assert failed.status == CollectionJobStatus.FAILED
    assert "broker down" in (failed.error_message or "")

    mock_start_job.side_effect = None
    retry = client.post(url, json=payload, headers=user_api_key_header)

    assert retry.status_code == 200
    assert retry.json()["data"]["job_id"] != str(failed.id)
