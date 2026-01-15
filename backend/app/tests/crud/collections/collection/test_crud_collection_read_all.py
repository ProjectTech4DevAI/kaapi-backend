from openai_responses import OpenAIMock
from openai import OpenAI
from sqlmodel import Session, delete

from app.crud import CollectionCrud
from app.models import Collection
from app.tests.utils.document import DocumentStore
from app.tests.utils.utils import get_project
from app.tests.utils.collection import get_collection


def create_collections(db: Session, n: int) -> Collection:
    crud = None
    project = get_project(db)
    openai_mock = OpenAIMock()
    with openai_mock.router:
        client = OpenAI(api_key="sk-test-key")
        for _ in range(n):
            collection = get_collection(db, project=project)
            store = DocumentStore(db, project_id=collection.project_id)
            documents = store.fill(1)
            if crud is None:
                crud = CollectionCrud(db, collection.project_id)
            crud.create(collection, documents)

        return crud.project_id


class TestCollectionReadAll:
    _ncollections = 5

    def test_number_read_is_expected(self, db: Session) -> None:
        db.exec(delete(Collection))

        owner = create_collections(db, self._ncollections)
        crud = CollectionCrud(db, owner)
        docs = crud.read_all()

        assert len(docs) == self._ncollections

    def test_deleted_docs_are_excluded(self, db: Session) -> None:
        owner = create_collections(db, self._ncollections)
        crud = CollectionCrud(db, owner)
        assert all(x.deleted_at is None for x in crud.read_all())
