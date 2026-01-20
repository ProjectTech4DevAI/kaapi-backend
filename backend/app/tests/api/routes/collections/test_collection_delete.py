from uuid import UUID, uuid4
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext
from app.models import CollectionJobStatus
from app.tests.utils.utils import get_project
from app.tests.utils.collection import get_collection


@patch("app.api.routes.collections.delete_service.start_job")
def test_delete_collection_calls_start_job_and_returns_job(
    mock_start_job: Any,
    db: Session,
    client: TestClient,
    user_api_key_header: dict[str, str],
    user_api_key: TestAuthContext,
) -> None:
    """
    Happy path:
    - Existing collection for the current project
    - No callback request body
    - Creates a DELETE CollectionJob with PENDING status
    - Calls delete_service.start_job with correct arguments
    """
    project = get_project(db, "Dalgo")
    collection = get_collection(db, project)

    resp = client.request(
        "DELETE",
        f"{settings.API_V1_STR}/collections/{collection.id}",
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

    returned_job_id = UUID(data["job_id"])
    assert kwargs["collection_job_id"] == returned_job_id

    deletion_request = kwargs["request"]
    assert deletion_request.collection_id == collection.id
    assert deletion_request.callback_url is None


@patch("app.api.routes.collections.delete_service.start_job")
def test_delete_collection_with_callback_url_passes_it_to_start_job(
    mock_start_job: Any,
    db: Session,
    client: TestClient,
    user_api_key_header: dict[str, str],
    user_api_key: TestAuthContext,
) -> None:
    """
    When a callback_url is provided in the request body, ensure it is passed
    into the DeletionRequest and then into delete_service.start_job.
    """
    project = get_project(db, "Dalgo")
    collection = get_collection(db, project)

    payload = {
        "callback_url": "https://example.com/collections/delete-callback",
    }

    resp = client.request(
        "DELETE",
        f"{settings.API_V1_STR}/collections/{collection.id}",
        json=payload,
        headers=user_api_key_header,
    )

    assert resp.status_code == 200
    body = resp.json()

    data = body["data"]
    assert data["status"] == CollectionJobStatus.PENDING

    mock_start_job.assert_called_once()
    kwargs = mock_start_job.call_args.kwargs

    assert kwargs["project_id"] == user_api_key.project_id
    assert kwargs["organization_id"] == user_api_key.organization_id
    assert kwargs["collection_job_id"] == UUID(data["job_id"])

    deletion_request = kwargs["request"]
    assert deletion_request.collection_id == collection.id
    assert str(deletion_request.callback_url) == payload["callback_url"]


@patch("app.api.routes.collections.delete_service.start_job")
def test_delete_collection_not_found_returns_404_and_does_not_start_job(
    mock_start_job: Any,
    client: TestClient,
    user_api_key_header: dict[str, str],
) -> None:
    """
    For a random UUID that doesn't correspond to any collection, we expect:
    - 404 response
    - delete_service.start_job is NOT called
    """
    random_id = uuid4()

    resp = client.request(
        "DELETE",
        f"{settings.API_V1_STR}/collections/{random_id}",
        headers=user_api_key_header,
    )

    assert resp.status_code == 404
    mock_start_job.assert_not_called()
