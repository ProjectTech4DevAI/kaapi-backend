import pytest
from sqlmodel import Session, select

from app.crud import DocumentCrud
from app.models import Document
from app.tests.utils.document import DocumentStore
from app.tests.utils.utils import get_project
from app.tests.utils.test_data import create_test_project
from app.core.exception_handlers import HTTPException


@pytest.fixture
def document(db: Session) -> Document:
    project = get_project(db)
    store = DocumentStore(db, project.id)
    document = store.put()

    crud = DocumentCrud(db, document.project_id)
    crud.delete(document.id)

    statement = select(Document).where(Document.id == document.id)
    return db.exec(statement).one()


class TestDatabaseDelete:
    def test_delete_is_soft(self, document: Document) -> None:
        assert document is not None

    def test_delete_marks_deleted(self, document: Document) -> None:
        assert document.is_deleted is True

    def test_delete_follows_insert(self, document: Document) -> None:
        assert document.inserted_at <= document.deleted_at

    def test_cannot_delete_others_documents(self, db: Session) -> None:
        project = get_project(db)
        store = DocumentStore(db, project.id)
        document = store.put()
        other_project = create_test_project(db)

        crud = DocumentCrud(db, other_project.id)
        with pytest.raises(HTTPException) as exc_info:
            crud.delete(document.id)

        assert exc_info.value.status_code == 404
        assert "Document not found" in str(exc_info.value.detail)
