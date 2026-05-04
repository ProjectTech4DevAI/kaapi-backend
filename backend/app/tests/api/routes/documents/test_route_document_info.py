import pytest
from sqlmodel import Session

from app.models.config.config import ConfigTag
from app.tests.utils.document import (
    DocumentComparator,
    DocumentMaker,
    DocumentStore,
    Route,
    WebCrawler,
    httpx_to_standard,
)


@pytest.fixture
def route():
    return Route("")


class TestDocumentRouteInfo:
    def test_response_is_success(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        response = crawler.get(route.append(store.put()))

        assert response.is_success

    def test_info_reflects_database(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        document = store.put()
        source = DocumentComparator(document)

        target = httpx_to_standard(crawler.get(route.append(document)))

        assert source == target.data

    def test_cannot_info_unknown_document(
        self, db: Session, route: Route, crawler: WebCrawler
    ) -> None:
        DocumentStore.clear(db)
        maker = DocumentMaker(project_id=crawler.user_api_key.project_id, session=db)
        response = crawler.get(route.append(next(maker)))

        assert response.is_error

    def test_explicit_default_tag_allows_default_document(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        document = store.put()

        response = crawler.get(Route("", tag=ConfigTag.DEFAULT.value).append(document))

        assert response.is_success

    def test_explicit_tag_allows_matching_document(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        document = store.put(tag=ConfigTag.DEFAULT)

        response = crawler.get(Route("", tag=ConfigTag.DEFAULT.value).append(document))

        assert response.is_success
