import pytest
from sqlmodel import Session

from app.models.config.config import ConfigTag
from app.tests.utils.document import (
    DocumentComparator,
    DocumentStore,
    Route,
    WebCrawler,
    httpx_to_standard,
)


class QueryRoute(Route):
    def pushq(self, key, value):
        qs_args = self.qs_args | {
            key: value,
        }
        return type(self)(self.endpoint, **qs_args)


@pytest.fixture
def route():
    return QueryRoute("")


class TestDocumentRouteList:
    _ndocs = 10

    def test_response_is_success(self, route: QueryRoute, crawler: WebCrawler) -> None:
        response = crawler.get(route)
        assert response.is_success

    def test_empty_db_returns_empty_list(
        self,
        db: Session,
        route: QueryRoute,
        crawler: WebCrawler,
    ) -> None:
        DocumentStore.clear(db)
        response = httpx_to_standard(crawler.get(route))

        assert not response.data

    def test_item_reflects_database(
        self,
        db: Session,
        route: QueryRoute,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        source = DocumentComparator(store.put())

        response = httpx_to_standard(crawler.get(route))
        (target,) = response.data

        assert source == target

    def test_no_tag_returns_default_documents(
        self,
        db: Session,
        route: QueryRoute,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        implicit_default_doc = store.put()
        default_doc = store.put(tag=ConfigTag.DEFAULT)

        response = httpx_to_standard(crawler.get(route))
        ids = [item["id"] for item in response.data]

        assert str(implicit_default_doc.id) in ids
        assert str(default_doc.id) in ids

    def test_explicit_tag_returns_matching_documents(
        self,
        db: Session,
        route: QueryRoute,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        implicit_default_doc = store.put()
        default_doc = store.put(tag=ConfigTag.DEFAULT)

        response = httpx_to_standard(crawler.get(route.pushq("tag", "default")))
        ids = [item["id"] for item in response.data]

        assert str(default_doc.id) in ids
        assert str(implicit_default_doc.id) in ids

    def test_negative_skip_produces_error(
        self,
        route: QueryRoute,
        crawler: WebCrawler,
    ) -> None:
        response = crawler.get(route.pushq("skip", -1))
        assert response.is_error

    def test_negative_limit_produces_error(
        self,
        route: QueryRoute,
        crawler: WebCrawler,
    ) -> None:
        response = crawler.get(route.pushq("limit", -1))
        assert response.is_error

    def test_skip_greater_than_limit_is_difference(
        self,
        db: Session,
        route: QueryRoute,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        limit = len(store.fill(self._ndocs))
        skip = limit // 2

        route = route.pushq("skip", skip).pushq("limit", limit)
        response = httpx_to_standard(crawler.get(route))

        assert len(response.data) == limit - skip
