import uuid
from typing import Optional

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.tests.utils.utils import get_project, get_document
from app.tests.utils.collection import get_collection, get_vector_store_collection
from app.crud import DocumentCollectionCrud
from app.models import Collection, Document


def link_document_to_collection(
    db: Session,
    collection: Collection,
    document: Optional[Document] = None,
) -> Document:
    """
    Utility used in tests to associate a Document with a Collection so that
    DocumentCollectionCrud.read(...) will return something.

    If you have not given documents to this function then this uses your `get_document` helper
    to provide documents to the DocumentCollectionCrud.create.
    """

    if document is None:
        document = get_document(db)

    crud = DocumentCollectionCrud(db)
    crud.create(collection, [document])

    return document


def test_collection_info_returns_assistant_collection_with_docs(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
) -> None:
    """
    Happy path:
    - Assistant-style collection (get_collection)
    - include_docs = True (default)
    - At least one document linked
    """

    project = get_project(db, "Dalgo")
    collection = get_collection(db, project)

    document = link_document_to_collection(db, collection)

    response = client.get(
        f"{settings.API_V1_STR}/collections/{collection.id}",
        headers=user_api_key_header,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    payload = data["data"]

    assert str(collection.id) == payload["id"]
    assert payload["project_id"] == project.id

    docs = payload.get("documents", [])
    assert isinstance(docs, list)
    assert len(docs) >= 1

    doc_ids = {d["id"] for d in docs}
    assert str(document.id) in doc_ids


def test_collection_info_include_docs_false_returns_no_docs(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
) -> None:
    """
    When include_docs=false, the endpoint should not populate the documents list.
    """
    project = get_project(db, "Dalgo")
    collection = get_collection(db, project)

    link_document_to_collection(db, collection)

    response = client.get(
        f"{settings.API_V1_STR}/collections/{collection.id}",
        headers=user_api_key_header,
        params={"include_docs": "false"},
    )

    assert response.status_code == 200

    data = response.json()
    payload = data["data"]

    assert payload["id"] == str(collection.id)
    assert payload["llm_service_name"] == "gpt-4o"
    assert payload["llm_service_id"] == collection.llm_service_id
    assert payload["documents"] is None


def test_collection_info_pagination_skip_and_limit(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
) -> None:
    """
    Verify skip & limit are passed through to DocumentCollectionCrud.read.
    We create multiple document links and then request a paginated slice.
    """
    project = get_project(db, "Dalgo")
    collection = get_collection(db, project)

    documents = db.exec(
        select(Document).where(Document.deleted_at.is_(None)).limit(2)
    ).all()

    assert len(documents) >= 2, "Test requires at least two documents in the DB"

    DocumentCollectionCrud(db).create(collection, documents)

    response = client.get(
        f"{settings.API_V1_STR}/collections/{collection.id}",
        headers=user_api_key_header,
        params={"skip": 1, "limit": 1},
    )

    assert response.status_code == 200

    data = response.json()
    payload = data["data"]
    docs_resp = payload.get("documents", [])

    assert len(docs_resp) == 1


def test_collection_info_vector_store_collection(
    client: TestClient,
    db: Session,
    user_api_key_header: dict[str, str],
) -> None:
    """
    Ensure the endpoint also works for vector-store-style collections created
    via get_vector_store_collection.
    """
    project = get_project(db, "Dalgo")
    collection = get_vector_store_collection(db, project)

    link_document_to_collection(db, collection)

    response = client.get(
        f"{settings.API_V1_STR}/collections/{collection.id}",
        headers=user_api_key_header,
    )

    assert response.status_code == 200

    data = response.json()
    payload = data["data"]

    assert payload["id"] == str(collection.id)
    assert payload["llm_service_name"] == "openai vector store"
    assert payload["llm_service_id"] == collection.llm_service_id

    docs = payload.get("documents", [])
    assert len(docs) >= 1


def test_collection_info_not_found_returns_404(
    client: TestClient,
    user_api_key_header: dict[str, str],
) -> None:
    """
    For a random UUID that doesn't correspond to any collection, we expect 404.
    """
    random_id = uuid.uuid4()

    response = client.get(
        f"{settings.API_V1_STR}/collections/{random_id}",
        headers=user_api_key_header,
    )

    assert response.status_code == 404


def test_collection_info_include_docs_and_url(
    client: TestClient,
    db: Session,
    user_api_key_header,
) -> None:
    """
    Test that when include_docs=true and include_url=true,
    the endpoint returns documents with their URLs.
    """
    project = get_project(db, "Dalgo")
    collection = get_collection(db, project)

    document = link_document_to_collection(db, collection)

    response = client.get(
        f"{settings.API_V1_STR}/collections/{collection.id}",
        headers=user_api_key_header,
        params={"include_docs": "true", "include_url": "true"},
    )

    assert response.status_code == 200

    data = response.json()
    payload = data["data"]

    assert payload["id"] == str(collection.id)

    docs = payload.get("documents", [])
    assert isinstance(docs, list)
    assert len(docs) >= 1

    doc_ids = {d["id"] for d in docs}
    assert str(document.id) in doc_ids

    doc = next(d for d in docs if d["id"] == str(document.id))
    assert "signed_url" in doc
    assert doc["signed_url"].startswith("https://")
