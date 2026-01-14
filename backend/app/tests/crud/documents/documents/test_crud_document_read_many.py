import pytest
from sqlmodel import Session

from app.crud import DocumentCrud
from app.tests.utils.document import DocumentStore
from app.tests.utils.utils import get_project


@pytest.fixture
def store(db: Session) -> DocumentStore:
    project = get_project(db)
    ds = DocumentStore(db, project.id)
    ds.fill(TestDatabaseReadMany._ndocs)

    return ds


class TestDatabaseReadMany:
    _ndocs = 10

    def test_number_read_is_expected(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        docs = crud.read_many()
        assert len(docs) == self._ndocs

    def test_deleted_docs_are_excluded(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        assert all(x.is_deleted is False for x in crud.read_many())

    def test_skip_is_respected(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        skip = self._ndocs // 2
        docs = crud.read_many(skip=skip)

        assert len(docs) == self._ndocs - skip

    def test_zero_skip_includes_all(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        docs = crud.read_many(skip=0)
        assert len(docs) == self._ndocs

    def test_big_skip_is_empty(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        skip = self._ndocs + 1
        assert not crud.read_many(skip=skip)

    def test_negative_skip_raises_exception(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        with pytest.raises(ValueError):
            crud.read_many(skip=-1)

    def test_limit_is_respected(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        limit = self._ndocs // 2
        docs = crud.read_many(limit=limit)

        assert len(docs) == limit

    def test_zero_limit_includes_nothing(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        assert not crud.read_many(limit=0)

    def test_negative_limit_raises_exception(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        with pytest.raises(ValueError):
            crud.read_many(limit=-1)

    def test_skip_greater_than_limit_is_difference(
        self,
        db: Session,
        store: DocumentStore,
    ) -> None:
        crud = DocumentCrud(db, store.project.id)
        limit = self._ndocs
        skip = limit // 2
        docs = crud.read_many(skip=skip, limit=limit)

        assert len(docs) == limit - skip
