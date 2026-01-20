import openai_responses
from openai import OpenAI
from sqlmodel import Session, select

from app.crud import CollectionCrud
from app.models import APIKey, Collection
from app.crud.rag import OpenAIAssistantCrud
from app.tests.utils.utils import get_project
from app.tests.utils.document import DocumentStore


def get_collection_for_delete(
    db: Session, client=None, project_id: int = None
) -> Collection:
    project = get_project(db)
    if client is None:
        client = OpenAI(api_key="test_api_key")

    vector_store = client.vector_stores.create()
    assistant = client.beta.assistants.create(
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    return Collection(
        organization_id=project.organization_id,
        project_id=project_id,
        llm_service_id=assistant.id,
        llm_service_name="gpt-4o",
    )


class TestCollectionDelete:
    _n_collections = 5

    @openai_responses.mock()
    def test_delete_marks_deleted(self, db: Session) -> None:
        project = get_project(db)
        client = OpenAI(api_key="sk-test-key")

        assistant = OpenAIAssistantCrud(client)
        collection = get_collection_for_delete(db, client, project_id=project.id)

        crud = CollectionCrud(db, collection.project_id)
        collection_ = crud.delete(collection, assistant)

        assert collection_.deleted_at is not None

    @openai_responses.mock()
    def test_delete_follows_insert(self, db: Session) -> None:
        client = OpenAI(api_key="sk-test-key")

        assistant = OpenAIAssistantCrud(client)
        project = get_project(db)
        collection = get_collection_for_delete(db, project_id=project.id)

        crud = CollectionCrud(db, collection.project_id)
        collection_ = crud.delete(collection, assistant)

        assert collection_.inserted_at <= collection_.deleted_at

    @openai_responses.mock()
    def test_delete_document_deletes_collections(self, db: Session) -> None:
        project = get_project(db)
        store = DocumentStore(db, project_id=project.id)
        documents = store.fill(1)

        stmt = select(APIKey).where(
            APIKey.project_id == project.id, APIKey.is_deleted == False
        )
        api_key = db.exec(stmt).first()

        client = OpenAI(api_key="sk-test-key")
        resources = []
        for _ in range(self._n_collections):
            coll = get_collection_for_delete(db, client, project_id=project.id)
            crud = CollectionCrud(db, project_id=project.id)
            collection = crud.create(coll, documents)
            resources.append((crud, collection))

        ((crud, _), *_) = resources
        assistant = OpenAIAssistantCrud(client)
        crud.delete(documents[0], assistant)

        assert all(y.deleted_at for (_, y) in resources)
