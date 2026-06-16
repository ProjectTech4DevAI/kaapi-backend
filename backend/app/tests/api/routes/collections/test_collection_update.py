from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.crud import CollectionCrud
from app.models import CollectionUpdate
from app.tests.utils.utils import get_project
from app.tests.utils.collection import get_vector_store_collection


def test_update_collection_returns_updated_fields(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
) -> None:
    project = get_project(db, "Dalgo")
    collection = get_vector_store_collection(db, project)

    response = client.patch(
        f"{settings.API_V1_STR}/collections/{collection.id}",
        headers=user_api_key_header,
        json={"name": "edited", "description": "edited desc"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["id"] == str(collection.id)
    assert data["name"] == "edited"
    assert data["description"] == "edited desc"


def test_update_collection_partial_update_preserves_other_fields(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
) -> None:
    project = get_project(db, "Dalgo")
    collection = get_vector_store_collection(db, project)
    CollectionCrud(db, project.id).update(
        collection.id, CollectionUpdate(name="original", description="original-desc")
    )

    response = client.patch(
        f"{settings.API_V1_STR}/collections/{collection.id}",
        headers=user_api_key_header,
        json={"description": "patched-desc"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["name"] == "original"
    assert data["description"] == "patched-desc"


def test_update_collection_rename_to_existing_name_returns_409(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
) -> None:
    project = get_project(db, "Dalgo")
    crud = CollectionCrud(db, project.id)

    first = get_vector_store_collection(db, project)
    crud.update(first.id, CollectionUpdate(name="duplicate"))

    second = get_vector_store_collection(db, project)

    response = client.patch(
        f"{settings.API_V1_STR}/collections/{second.id}",
        headers=user_api_key_header,
        json={"name": "duplicate"},
    )

    assert response.status_code == 409


def test_update_collection_not_found_returns_404(
    client: TestClient,
    user_api_key_header: dict[str, str],
) -> None:
    response = client.patch(
        f"{settings.API_V1_STR}/collections/{uuid4()}",
        headers=user_api_key_header,
        json={"name": "irrelevant"},
    )

    assert response.status_code == 404
