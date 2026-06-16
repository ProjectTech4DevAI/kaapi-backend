import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.crud import CollectionCrud
from app.models import Collection
from app.tests.utils.document import DocumentStore
from app.tests.utils.utils import get_project
from app.tests.utils.collection import get_vector_store_collection


def mk_collection(db: Session) -> Collection:
    project = get_project(db)
    collection = get_vector_store_collection(db, project=project)
    store = DocumentStore(db, project_id=collection.project_id)
    documents = store.fill(1)
    crud = CollectionCrud(db, collection.project_id)
    return crud.create(collection, documents)


class TestDatabaseReadOne:
    def test_can_select_valid_id(self, db: Session) -> None:
        collection = mk_collection(db)

        crud = CollectionCrud(db, collection.project_id)
        result = crud.read_one(collection.id)

        assert result.id == collection.id

    def test_cannot_select_others_collections(self, db: Session) -> None:
        collection = mk_collection(db)
        other = collection.project_id + 1
        crud = CollectionCrud(db, other)
        with pytest.raises(HTTPException) as excinfo:
            crud.read_one(collection.id)
        assert excinfo.value.status_code == 404
        assert excinfo.value.detail == "Collection not found"
