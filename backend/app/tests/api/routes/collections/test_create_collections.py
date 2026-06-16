from uuid import UUID, uuid4
from unittest.mock import patch
from typing import Any

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext
from app.models import CollectionJobStatus, Document
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
