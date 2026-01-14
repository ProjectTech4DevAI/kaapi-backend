import pytest
from sqlmodel import Session

from app.crud import DocumentCrud
from app.tests.utils.document import DocumentMaker, DocumentStore
from app.tests.utils.utils import get_project
from app.tests.utils.test_data import create_test_project


@pytest.fixture
def documents(db: Session) -> DocumentMaker:
    project = get_project(db)
    store = DocumentStore(db, project.id)
    return store.documents


class TestDatabaseUpdate:
    def test_update_adds_one(self, db: Session, documents: DocumentMaker) -> None:
        crud = DocumentCrud(db, documents.project_id)

        before = crud.read_many()
        crud.update(next(documents))
        after = crud.read_many()

        assert len(before) + 1 == len(after)

    def test_sequential_update_is_ordered(
        self,
        db: Session,
        documents: DocumentMaker,
    ) -> None:
        crud = DocumentCrud(db, documents.project_id)
        (a, b) = (crud.update(y) for (_, y) in zip(range(2), documents))

        assert a.inserted_at <= b.inserted_at

    def test_insert_does_not_delete(
        self,
        db: Session,
        documents: DocumentMaker,
    ) -> None:
        crud = DocumentCrud(db, documents.project_id)
        document = crud.update(next(documents))

        assert document.is_deleted is False

    def test_update_sets_default_owner(
        self,
        db: Session,
        documents: DocumentMaker,
    ) -> None:
        crud = DocumentCrud(db, documents.project_id)
        document = next(documents)
        document.project_id = None
        result = crud.update(document)

        assert result.project_id == documents.project_id

    def test_update_respects_owner(
        self,
        db: Session,
        documents: DocumentMaker,
    ) -> None:
        document = next(documents)
        other_project = create_test_project(db)
        document.project_id = other_project.id

        crud = DocumentCrud(db, documents.project_id)
        with pytest.raises(PermissionError):
            crud.update(document)
