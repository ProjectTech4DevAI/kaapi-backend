from unittest.mock import patch
from typing import Any

import pytest
import openai_responses
from openai_responses import OpenAIMock
from openai import OpenAI
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
    @openai_responses.mock()
    @patch("app.api.routes.documents.get_openai_client")
    def test_response_is_success(
        self,
        mock_get_openai_client: Any,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        openai_mock = OpenAIMock()
        with openai_mock.router:
            client = OpenAI(api_key="sk-test-key")
            mock_get_openai_client.return_value = client

            store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
            response = crawler.delete(route.append(store.put()))

            assert response.is_success

    @openai_responses.mock()
    @patch("app.api.routes.documents.get_openai_client")
    def test_item_is_soft_removed(
        self,
        mock_get_openai_client: Any,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        openai_mock = OpenAIMock()
        with openai_mock.router:
            client = OpenAI(api_key="sk-test-key")
            mock_get_openai_client.return_value = client

            store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
            document = store.put()

            crawler.delete(route.append(document))
            db.refresh(document)
            statement = select(Document).where(Document.id == document.id)
            result = db.exec(statement).one()

            assert result.is_deleted is True

    @openai_responses.mock()
    @patch("app.api.routes.documents.get_openai_client")
    def test_cannot_remove_unknown_document(
        self,
        mock_get_openai_client: Any,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        openai_mock = OpenAIMock()
        with openai_mock.router:
            client = OpenAI(api_key="sk-test-key")
            mock_get_openai_client.return_value = client

            DocumentStore.clear(db)

            maker = DocumentMaker(
                project_id=crawler.user_api_key.project_id, session=db
            )
            response = crawler.delete(route.append(next(maker)))

            assert response.is_error
