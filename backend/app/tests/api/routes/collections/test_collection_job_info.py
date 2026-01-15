from typing import Any

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.tests.utils.utils import get_project
from app.tests.utils.collection import get_collection, get_collection_job
from app.models import (
    CollectionActionType,
    CollectionJobStatus,
)


def test_collection_info_processing(
    db: Session, client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    headers = user_api_key_header
    project = get_project(db, "Dalgo")

    collection_job = get_collection_job(
        db, project, status=CollectionJobStatus.PROCESSING
    )

    resp = client.get(
        f"{settings.API_V1_STR}/collections/jobs/{collection_job.id}",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()["data"]

    assert data["job_id"] == str(collection_job.id)
    assert data["status"] == CollectionJobStatus.PROCESSING

    assert data.get("collection") is None


def test_collection_info_create_successful(
    db: Session, client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    headers = user_api_key_header
    project = get_project(db, "Dalgo")

    collection = get_collection(db, project)

    collection_job = get_collection_job(
        db, project, collection_id=collection.id, status=CollectionJobStatus.SUCCESSFUL
    )

    resp = client.get(
        f"{settings.API_V1_STR}/collections/jobs/{collection_job.id}",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()["data"]

    assert data["job_id"] == str(collection_job.id)
    assert data["status"] == CollectionJobStatus.SUCCESSFUL
    assert data["action_type"] == CollectionActionType.CREATE

    assert data["collection"] is not None
    col = data["collection"]
    assert col["id"] == str(collection.id)
    assert col["llm_service_id"] == collection.llm_service_id
    assert col["llm_service_name"] == "gpt-4o"


def test_collection_info_create_failed(
    db: Session, client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    headers = user_api_key_header
    project = get_project(db, "Dalgo")

    collection_job = get_collection_job(
        db,
        project,
        status=CollectionJobStatus.FAILED,
        error_message="something went wrong",
    )

    resp = client.get(
        f"{settings.API_V1_STR}/collections/jobs/{collection_job.id}",
        headers=headers,
    )
    body = resp.json()
    assert body["success"] is True

    data = body["data"]

    assert data["job_id"] == str(collection_job.id)
    assert data["status"] == CollectionJobStatus.FAILED
    assert data["action_type"] == CollectionActionType.CREATE
    assert data["error_message"] == "something went wrong"

    assert data["collection"] is None


def test_collection_info_delete_successful(
    db: Session, client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    headers = user_api_key_header
    project = get_project(db, "Dalgo")

    collection = get_collection(db, project)

    collection_job = get_collection_job(
        db,
        project,
        collection_id=collection.id,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.SUCCESSFUL,
    )

    resp = client.get(
        f"{settings.API_V1_STR}/collections/jobs/{collection_job.id}",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()["data"]

    assert data["job_id"] == str(collection_job.id)
    assert data["status"] == CollectionJobStatus.SUCCESSFUL
    assert data["action_type"] == CollectionActionType.DELETE

    assert data["collection"] is not None
    col = data["collection"]
    assert col["id"] == str(collection.id)


def test_collection_info_delete_failed(
    db: Session, client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    headers = user_api_key_header
    project = get_project(db, "Dalgo")

    collection = get_collection(db, project)

    collection_job = get_collection_job(
        db,
        project,
        collection_id=collection.id,
        action_type=CollectionActionType.DELETE,
        status=CollectionJobStatus.FAILED,
        error_message="something went wrong",
    )

    resp = client.get(
        f"{settings.API_V1_STR}/collections/jobs/{collection_job.id}",
        headers=headers,
    )
    body = resp.json()
    assert body["success"] is True

    data = body["data"]
    assert data["job_id"] == str(collection_job.id)
    assert data["status"] == CollectionJobStatus.FAILED
    assert data["action_type"] == CollectionActionType.DELETE
    assert data["error_message"] == "something went wrong"

    assert data["collection"] is not None
