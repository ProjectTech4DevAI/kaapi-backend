from uuid import UUID, uuid4
from unittest.mock import patch
from typing import Any

from fastapi.testclient import TestClient

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext
from app.models import CollectionJobStatus
from app.models.collection import CreationRequest


def _extract_metadata(body: dict) -> dict | None:
    return body.get("metadata") or body.get("meta")


@patch("app.api.routes.collections.create_service.start_job")
def test_collection_creation_with_assistant_calls_start_job_and_returns_job(
    mock_start_job: Any,
    client: TestClient,
    user_api_key_header: dict[str, str],
    user_api_key: TestAuthContext,
) -> None:
    creation_data = CreationRequest(
        model="gpt-4o",
        instructions="string",
        temperature=0.000001,
        documents=[UUID("f3e86a17-1e6f-41ec-b020-5b08eebef928")],
        batch_size=1,
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

    assert _extract_metadata(body) in (None, {})

    mock_start_job.assert_called_once()
    kwargs = mock_start_job.call_args.kwargs
    assert "db" in kwargs
    assert kwargs["project_id"] == user_api_key.project_id
    assert kwargs["organization_id"] == user_api_key.organization_id
    assert kwargs["with_assistant"] is True

    returned_job_id = UUID(data["job_id"])
    assert kwargs["collection_job_id"] == returned_job_id

    assert kwargs["request"].model_dump(mode="json") == creation_data.model_dump(
        mode="json"
    )


@patch("app.api.routes.collections.create_service.start_job")
def test_collection_creation_vector_only_adds_metadata_and_sets_with_assistant_false(
    mock_start_job: Any,
    client: TestClient,
    user_api_key_header: dict[str, str],
    user_api_key: TestAuthContext,
) -> None:
    creation_data = CreationRequest(
        temperature=0.000001,
        documents=[str(uuid4())],
        batch_size=1,
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

    meta = _extract_metadata(body)
    assert isinstance(meta, dict)
    assert "vector store only" in meta.get("note", "").lower()

    mock_start_job.assert_called_once()
    kwargs = mock_start_job.call_args.kwargs
    assert kwargs["project_id"] == user_api_key.project_id
    assert kwargs["organization_id"] == user_api_key.organization_id
    assert kwargs["with_assistant"] is False
    assert kwargs["collection_job_id"] == UUID(data["job_id"])
    assert kwargs["request"].model_dump(mode="json") == creation_data.model_dump(
        mode="json"
    )


def test_collection_creation_vector_only_request_validation_error(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    payload = {
        "model": "gpt-4o",
        "temperature": 0.000001,
        "documents": [str(uuid4())],
        "batch_size": 1,
        "callback_url": None,
    }

    resp = client.post(
        f"{settings.API_V1_STR}/collections",
        json=payload,
        headers=user_api_key_header,
    )

    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["metadata"] is None
    assert (
        "To create an Assistant, provide BOTH 'model' and 'instructions'"
        in body["error"]
    )
