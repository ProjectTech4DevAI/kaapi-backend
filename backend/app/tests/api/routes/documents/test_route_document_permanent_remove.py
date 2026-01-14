import os
from pathlib import Path
from urllib.parse import urlparse
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from moto import mock_aws
from sqlmodel import Session, select

from openai import OpenAI
import openai_responses
from openai_responses import OpenAIMock

from app.core.cloud import AmazonCloudStorageClient
from app.core.config import settings
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


@pytest.fixture(scope="class")
def aws_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION


@pytest.mark.usefixtures("aws_credentials")
@mock_aws
class TestDocumentRoutePermanentRemove:
    @openai_responses.mock()
    @patch("app.api.routes.documents.get_openai_client")
    def test_permanent_delete_document_from_s3(
        self,
        mock_get_openai_client,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        openai_mock = OpenAIMock()
        with openai_mock.router:
            client = OpenAI(api_key="sk-test-key")
            mock_get_openai_client.return_value = client

        # Setup AWS
        aws = AmazonCloudStorageClient()
        aws.create()

        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        document = store.put()
        s3_key = Path(urlparse(document.object_store_url).path).relative_to("/")
        aws.client.put_object(
            Bucket=settings.AWS_S3_BUCKET, Key=str(s3_key), Body=b"test"
        )

        # Delete document
        response = crawler.delete(route.append(document, suffix="permanent"))
        assert response.is_success

        db.refresh(document)

        stmt = select(Document).where(Document.id == document.id)
        doc_in_db = db.exec(stmt).first()
        assert doc_in_db is not None
        assert doc_in_db.deleted_at is not None

        with pytest.raises(ClientError) as exc_info:
            aws.client.head_object(
                Bucket=settings.AWS_S3_BUCKET,
                Key=str(s3_key),
            )
        assert exc_info.value.response["Error"]["Code"] == "404"

    @openai_responses.mock()
    def test_cannot_delete_nonexistent_document(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        DocumentStore.clear(db)

        maker = DocumentMaker(project_id=crawler.user_api_key.project_id, session=db)
        response = crawler.delete(route.append(next(maker), suffix="permanent"))

        assert response.is_error
