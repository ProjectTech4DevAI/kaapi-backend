from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlmodel import Session

from app.core.cloud import AmazonCloudStorageClient
from app.core.config import settings
from app.models import Document
from app.tests.utils.auth import TestAuthContext

UPLOAD_URL_ROUTE = f"{settings.API_V2_STR}/documents/upload-url"


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestDocumentUploadURLV2:
    def test_returns_presigned_put_url_for_new_document_id(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()

        response = client.post(
            UPLOAD_URL_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "quarterly-report.pdf"},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["expires_in"] == 3600

        document_id = UUID(data["document_id"])
        key = f"{user_api_key.project.storage_path}/{document_id}"
        assert key in data["upload_url"]
        assert "X-Amz-Signature" in data["upload_url"]

    def test_does_not_create_document_row(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()

        response = client.post(
            UPLOAD_URL_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "notes.txt"},
        )

        document_id = UUID(response.json()["data"]["document_id"])
        assert db.get(Document, document_id) is None

    def test_unsupported_extension_is_rejected(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        response = client.post(
            UPLOAD_URL_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "notes.xyz"},
        )

        assert response.status_code == 400
        assert "Unsupported file extension: .xyz" in response.json()["error"]

    def test_blank_filename_is_rejected(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        response = client.post(
            UPLOAD_URL_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": ""},
        )

        assert response.status_code == 422

    def test_missing_api_key_is_unauthorized(self, client: TestClient) -> None:
        response = client.post(UPLOAD_URL_ROUTE, json={"filename": "notes.txt"})

        assert response.status_code == 401

    def test_invalid_api_key_is_unauthorized(self, client: TestClient) -> None:
        response = client.post(
            UPLOAD_URL_ROUTE,
            headers={"X-API-KEY": "ApiKey not-a-real-key"},
            json={"filename": "notes.txt"},
        )

        assert response.status_code == 401
