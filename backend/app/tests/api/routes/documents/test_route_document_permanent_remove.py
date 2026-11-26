import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from botocore.exceptions import ClientError
from moto import mock_aws
from sqlmodel import Session

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


def _put_s3_object(aws: AmazonCloudStorageClient, document: Document) -> str:
    s3_key = str(Path(urlparse(document.object_store_url).path).relative_to("/"))
    aws.client.put_object(Bucket=settings.AWS_S3_BUCKET, Key=s3_key, Body=b"test")
    return s3_key


@pytest.mark.usefixtures("aws_credentials")
@mock_aws
class TestDocumentRoutePermanentRemove:
    def test_permanent_delete_removes_row_and_s3_object(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        aws = AmazonCloudStorageClient()
        aws.create()

        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        document = store.put()
        document_id = document.id
        s3_key = _put_s3_object(aws, document)

        response = crawler.delete(route.append(document, suffix="permanent"))
        assert response.is_success

        db.expire_all()
        assert db.get(Document, document_id) is None

        with pytest.raises(ClientError) as exc_info:
            aws.client.head_object(Bucket=settings.AWS_S3_BUCKET, Key=s3_key)
        assert exc_info.value.response["Error"]["Code"] == "404"

    def test_permanent_delete_cascades_to_transformed_documents(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        aws = AmazonCloudStorageClient()
        aws.create()

        project_id = crawler.user_api_key.project_id
        store = DocumentStore(db=db, project_id=project_id)
        source = store.put()
        _put_s3_object(aws, source)

        child = next(store.documents)
        child.source_document_id = source.id
        db.add(child)
        db.commit()
        child_id = child.id

        response = crawler.delete(route.append(source, suffix="permanent"))
        assert response.is_success

        db.expire_all()
        assert db.get(Document, child_id) is None

    def test_cannot_permanently_delete_soft_deleted_document(
        self,
        db: Session,
        route: Route,
        crawler: WebCrawler,
    ) -> None:
        store = DocumentStore(db=db, project_id=crawler.user_api_key.project_id)
        document = store.put()

        crawler.delete(route.append(document))
        response = crawler.delete(route.append(document, suffix="permanent"))

        assert response.status_code == 404
        db.expire_all()
        assert db.get(Document, document.id) is not None

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
