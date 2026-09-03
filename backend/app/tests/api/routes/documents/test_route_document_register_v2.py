from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
import requests
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from httpx import Response
from moto import mock_aws
from sqlmodel import Session

from app.core.cloud import AmazonCloudStorageClient
from app.core.cloud.storage import AmazonCloudStorage
from app.core.config import settings
from app.core.util import now
from app.models import Document
from app.services.collections.helpers import MAX_DOC_SIZE_MB
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.document import DocumentMaker

REGISTER_ROUTE = f"{settings.API_V2_STR}/documents"
UPLOAD_URL_ROUTE = f"{settings.API_V2_STR}/documents/upload-url"


def object_key(auth: TestAuthContext, document_id: UUID) -> str:
    return f"{auth.project.storage_path}/{document_id}"


def put_object(key: str, body: bytes) -> None:
    AmazonCloudStorageClient().client.put_object(
        Bucket=settings.AWS_S3_BUCKET, Key=key, Body=body
    )


def register(
    client: TestClient, auth: TestAuthContext, document_id: UUID, filename: str
) -> Response:
    return client.post(
        REGISTER_ROUTE,
        headers={"X-API-KEY": auth.key},
        json={"document_id": str(document_id), "filename": filename},
    )


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestDocumentRegisterV2:
    def test_registers_uploaded_object(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()
        document_id = uuid4()
        key = object_key(user_api_key, document_id)
        put_object(key, b"x" * 2048)

        response = register(client, user_api_key, document_id, "report.pdf")

        assert response.status_code == 201
        data = response.json()["data"]
        assert data["id"] == str(document_id)
        assert data["fname"] == "report.pdf"
        assert data["transformation_job"] is None
        assert data["signed_url"]

        document = db.get(Document, document_id)
        assert document is not None
        assert document.fname == "report.pdf"
        assert document.file_size_kb == 2.0
        assert document.object_store_url == f"s3://{settings.AWS_S3_BUCKET}/{key}"
        assert document.project_id == user_api_key.project_id

    def test_missing_object_is_rejected(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()
        document_id = uuid4()

        response = register(client, user_api_key, document_id, "report.pdf")

        assert response.status_code == 400
        assert "No uploaded file found" in response.json()["error"]
        assert db.get(Document, document_id) is None

    def test_oversized_object_is_rejected_and_deleted(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        aws = AmazonCloudStorageClient()
        aws.create()
        document_id = uuid4()
        key = object_key(user_api_key, document_id)
        put_object(key, b"x" * 1024)

        # Faking the reported size keeps a >25 MB body out of the test.
        oversized_kb = (MAX_DOC_SIZE_MB + 1) * 1024
        with patch.object(
            AmazonCloudStorage, "get_file_size_kb", return_value=oversized_kb
        ):
            response = register(client, user_api_key, document_id, "report.pdf")

        assert response.status_code == 413
        assert "exceeds the maximum allowed size" in response.json()["error"]
        assert db.get(Document, document_id) is None

        with pytest.raises(ClientError) as excinfo:
            aws.client.head_object(Bucket=settings.AWS_S3_BUCKET, Key=key)
        assert excinfo.value.response["Error"]["Code"] == "404"

    def test_duplicate_document_id_is_rejected(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()
        existing = next(DocumentMaker(project_id=user_api_key.project_id, session=db))
        db.add(existing)
        db.commit()
        put_object(object_key(user_api_key, existing.id), b"x" * 1024)

        response = register(client, user_api_key, existing.id, "report.pdf")

        assert response.status_code == 409
        assert "already registered" in response.json()["error"]

        db.refresh(existing)
        assert existing.fname != "report.pdf"

    def test_soft_deleted_document_id_is_rejected(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()
        deleted = next(DocumentMaker(project_id=user_api_key.project_id, session=db))
        deleted.deleted_at = now()
        db.add(deleted)
        db.commit()

        response = register(client, user_api_key, deleted.id, "report.pdf")

        assert response.status_code == 409
        assert "already registered" in response.json()["error"]

    def test_unsupported_extension_is_rejected(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()
        document_id = uuid4()
        put_object(object_key(user_api_key, document_id), b"x" * 1024)

        response = register(client, user_api_key, document_id, "report.xyz")

        assert response.status_code == 400
        assert "Unsupported file extension: .xyz" in response.json()["error"]
        assert db.get(Document, document_id) is None

    def test_missing_api_key_is_unauthorized(self, client: TestClient) -> None:
        response = client.post(
            REGISTER_ROUTE,
            json={"document_id": str(uuid4()), "filename": "report.pdf"},
        )

        assert response.status_code == 401


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestDocumentUploadRoundTripV2:
    def test_upload_url_then_put_then_register(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()

        url_response = client.post(
            UPLOAD_URL_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "handbook.pdf"},
        )
        assert url_response.status_code == 200
        upload_url = url_response.json()["data"]["upload_url"]
        document_id = UUID(url_response.json()["data"]["document_id"])

        put_response = requests.put(upload_url, data=b"y" * 3072)
        assert put_response.status_code == 200

        response = register(client, user_api_key, document_id, "handbook.pdf")

        assert response.status_code == 201
        assert response.json()["data"]["id"] == str(document_id)

        document = db.get(Document, document_id)
        assert document is not None
        assert document.file_size_kb == 3.0
        assert document.object_store_url == (
            f"s3://{settings.AWS_S3_BUCKET}/{object_key(user_api_key, document_id)}"
        )
