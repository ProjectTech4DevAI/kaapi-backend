import pytest
from sqlmodel import Session

from app.core.exception_handlers import HTTPException
from app.crud import DocumentCrud
from app.models.config.config import ConfigTag
from app.tests.utils.document import DocumentStore
from app.tests.utils.test_data import create_test_project
from app.tests.utils.utils import get_project


@pytest.fixture
def store(db: Session) -> DocumentStore:
    project = get_project(db)
    return DocumentStore(db, project.id)


class TestDatabaseReadOne:
    def test_can_select_valid_id(self, db: Session, store: DocumentStore) -> None:
        document = store.put()

        crud = DocumentCrud(db, store.project.id)
        result = crud.read_one(document.id)

        assert result.id == document.id

    def test_cannot_select_invalid_id(self, db: Session, store: DocumentStore) -> None:
        document = next(store.documents)

        crud = DocumentCrud(db, store.project.id)

        with pytest.raises(HTTPException) as exc_info:
            crud.read_one(document.id)

        assert exc_info.value.status_code == 404
        assert "Document not found" in str(exc_info.value.detail)

    def test_cannot_read_others_documents(
        self, db: Session, store: DocumentStore
    ) -> None:
        document = store.put()
        other_project = create_test_project(db)

        crud = DocumentCrud(db, other_project.id)
        with pytest.raises(HTTPException) as exc_info:
            crud.read_one(document.id)

        assert exc_info.value.status_code == 404
        assert "Document not found" in str(exc_info.value.detail)

    def test_no_tag_scope_allows_default_document(
        self, db: Session, store: DocumentStore
    ) -> None:
        document = store.put()

        crud = DocumentCrud(db, store.project.id)
        result = crud.read_one(document.id, tag=None)

        assert result.id == document.id

    def test_explicit_default_tag_scope_allows_default_document(
        self, db: Session, store: DocumentStore
    ) -> None:
        document = store.put()

        crud = DocumentCrud(db, store.project.id)
        result = crud.read_one(document.id, tag=ConfigTag.DEFAULT)

        assert result.id == document.id
