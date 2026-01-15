from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.tests.utils.utils import get_project
from app.tests.utils.collection import (
    get_collection,
    get_vector_store_collection,
)


def test_list_collections_returns_api_response(
    client: TestClient,
    user_api_key_header: dict[str, str],
) -> None:
    """
    Basic sanity check:
    - Endpoint returns 200
    - Response is wrapped in APIResponse
    - `data` is a list
    """

    response = client.get(
        f"{settings.API_V1_STR}/collections/",
        headers=user_api_key_header,
    )

    assert response.status_code == 200

    data = response.json()
    assert "success" in data
    assert "data" in data
    assert isinstance(data["data"], list)


def test_list_collections_includes_assistant_collection(
    db: Session,
    client: TestClient,
    user_api_key_header: dict[str, str],
) -> None:
    """
    Ensure that a newly created assistant-style collection (get_collection)
    appears in the list for the current project.
    """

    project = get_project(db, "Dalgo")

    response_before = client.get(
        f"{settings.API_V1_STR}/collections/",
        headers=user_api_key_header,
    )
    assert response_before.status_code == 200

    collection = get_collection(db, project)

    response_after = client.get(
        f"{settings.API_V1_STR}/collections/",
        headers=user_api_key_header,
    )
    assert response_after.status_code == 200

    after_data = response_after.json()
    assert after_data["success"] is True
    after_payload = after_data["data"]

    assert isinstance(after_payload, list)

    after_ids = {c["id"] for c in after_payload}
    assert str(collection.id) in after_ids

    for row in after_payload:
        assert row["project_id"] == project.id


def test_list_collections_includes_vector_store_collection_with_fields(
    db: Session,
    client: TestClient,
    user_api_key_header: dict[str, str],
) -> None:
    """
    Ensure that vector-store-style collections created via get_vector_store_collection
    appear in the list and expose the expected LLM fields.
    """
    project = get_project(db, "Dalgo")
    collection = get_vector_store_collection(db, project)

    response = client.get(
        f"{settings.API_V1_STR}/collections/",
        headers=user_api_key_header,
    )
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True

    rows = data["data"]
    assert isinstance(rows, list)

    matching = [c for c in rows if c["id"] == str(collection.id)]
    assert matching

    row = matching[0]
    assert row["project_id"] == project.id
    assert row["llm_service_name"] == "openai vector store"
    assert row["llm_service_id"] == collection.llm_service_id


def test_list_collections_does_not_error_with_no_collections(
    db: Session, client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    """
    If the project has no collections yet, the endpoint should still return
    200 and an empty list (or at least a list).
    This assumes a clean DB or that there may be zero collections initially.
    """
    response = client.get(
        f"{settings.API_V1_STR}/collections/",
        headers=user_api_key_header,
    )

    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert isinstance(data["data"], list)
