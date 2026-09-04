from unittest.mock import patch

import pytest
from sqlmodel import Session

from app.models import Document
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


class TestDocumentRouteInfoSignedUrl:
    """The `is_downloadable` flag decides whether the signed URL forces a save."""

    @staticmethod
    def _sign_and_capture(
        db: Session, route: Route, crawler: WebCrawler
    ) -> tuple[Document, list[str | None]]:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        document = store.put()
        with patch("app.api.routes.documents.get_cloud_storage") as mock_storage:
            mock_storage.return_value.get_signed_url.return_value = "https://signed"
            response = crawler.get(route.append(document))
            assert response.is_success
            filenames = [
                call.kwargs.get("filename")
                for call in mock_storage.return_value.get_signed_url.call_args_list
            ]
        return document, filenames

    def test_is_downloadable_true_signs_with_filename(
        self, db: Session, crawler: WebCrawler
    ) -> None:
        route = Route("", include_url="true", is_downloadable="true")

        document, filenames = self._sign_and_capture(db, route, crawler)

        assert filenames == [document.fname]

    def test_is_downloadable_omitted_signs_without_filename(
        self, db: Session, crawler: WebCrawler
    ) -> None:
        route = Route("", include_url="true")

        _, filenames = self._sign_and_capture(db, route, crawler)

        assert filenames == [None]
