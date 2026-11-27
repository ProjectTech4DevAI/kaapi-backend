import pytest
from sqlmodel import Session, select

from app.models import Document
from app.tests.utils.document import (
    DocumentMaker,
    DocumentStore,
    Route,
    WebCrawler,
)


@pytest.fixture
def route():
    return Route("")


class TestDocumentRouteRemove:
    def test_response_is_success(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        response = crawler.delete(route.append(store.put()))

        assert response.is_success

    def test_item_is_soft_removed(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        document = store.put()

        crawler.delete(route.append(document))
        db.refresh(document)
        statement = select(Document).where(Document.id == document.id)
        result = db.exec(statement).one()

        assert result.deleted_at is not None

    def test_cannot_remove_unknown_document(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        DocumentStore.clear(db)

        maker = DocumentMaker(project_id=crawler.user_api_key.project_id, session=db)
        response = crawler.delete(route.append(next(maker)))

        assert response.is_error

    def test_cannot_remove_already_deleted_document(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        document = store.put()

        crawler.delete(route.append(document))
        response = crawler.delete(route.append(document))

        assert response.status_code == 404
