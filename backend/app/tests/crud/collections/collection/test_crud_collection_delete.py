import openai_responses
from openai import OpenAI
from sqlmodel import Session, select

from app.crud import CollectionCrud
from app.models import APIKey, Collection, ProviderType
from app.crud.rag import OpenAIVectorStoreCrud
from app.tests.utils.utils import get_project
from app.tests.utils.document import DocumentStore


def get_vector_store_collection(
    db: Session, client: OpenAI, project_id: int
) -> Collection:
    vector_store = client.vector_stores.create()
    return Collection(
        project_id=project_id,
        llm_service_id=vector_store.id,
        llm_service_name="openai vector store",
        provider=ProviderType.openai,
    )


class TestCollectionDelete:
    _n_collections = 5

    @openai_responses.mock()
    def test_delete_marks_deleted(self, db: Session) -> None:
        project = get_project(db)
        client = OpenAI(api_key="sk-test-key")

        v_crud = OpenAIVectorStoreCrud(client)
        collection = get_vector_store_collection(db, client, project_id=project.id)

        crud = CollectionCrud(db, collection.project_id)
        collection_ = crud.delete(collection, v_crud)

        assert collection_.deleted_at is not None

    @openai_responses.mock()
    def test_delete_follows_insert(self, db: Session) -> None:
        project = get_project(db)
        client = OpenAI(api_key="sk-test-key")

        v_crud = OpenAIVectorStoreCrud(client)
        collection = get_vector_store_collection(db, client, project_id=project.id)

        crud = CollectionCrud(db, collection.project_id)
        collection_ = crud.delete(collection, v_crud)

        assert collection_.inserted_at <= collection_.deleted_at

    @openai_responses.mock()
    def test_delete_document_deletes_collections(self, db: Session) -> None:
        project = get_project(db)
        store = DocumentStore(db, project_id=project.id)
        documents = store.fill(1)

        stmt = select(APIKey).where(
            APIKey.project_id == project.id, APIKey.deleted_at.is_(None)
        )
        api_key = db.exec(stmt).first()

        client = OpenAI(api_key="sk-test-key")
        resources = []
        for _ in range(self._n_collections):
            coll = get_vector_store_collection(db, client, project_id=project.id)
            crud = CollectionCrud(db, project_id=project.id)
            collection = crud.create(coll, documents)
            resources.append((crud, collection))

        ((crud, _), *_) = resources
        v_crud = OpenAIVectorStoreCrud(client)
        crud.delete(documents[0], v_crud)

        assert all(y.deleted_at for (_, y) in resources)
