from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlmodel import Session

from app.core.cloud import AmazonCloudStorageClient
from app.core.config import settings
from app.models import Document
from app.tests.utils.auth import TestAuthContext

UPLOADS_ROUTE = f"{settings.API_V2_STR}/documents/uploads"


def pending_key(auth: TestAuthContext, document_id: str) -> str:
    # No extension: the filename lives in signed metadata, not in the key.
    return f"pending/{auth.project.storage_path}/{document_id}"


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestDocumentUploadsV2:
    def test_returns_presigned_post_for_pending_key(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()

        response = client.post(
            UPLOADS_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "quarterly-report.pdf"},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["expires_in"] == 3600

        fields = data["upload_fields"]
        assert fields["key"] == pending_key(user_api_key, data["document_id"])
        assert "x-amz-signature" in fields
        assert "register" in response.json()["metadata"]["next_step"]

    def test_upload_target_is_the_pending_key_not_the_final_one(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()

        response = client.post(
            UPLOADS_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "quarterly-report.pdf"},
        )

        data = response.json()["data"]
        key = data["upload_fields"]["key"]
        assert key == pending_key(user_api_key, data["document_id"])
        assert key != f"{user_api_key.project.storage_path}/{data['document_id']}"

    def test_filename_is_pinned_in_signed_metadata(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()

        response = client.post(
            UPLOADS_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "Quarterly Report.pdf"},
        )

        # Signed as a field so the client cannot change it; URL-encoded for the header.
        fields = response.json()["data"]["upload_fields"]
        assert fields["x-amz-meta-filename"] == quote("Quarterly Report.pdf")

    def test_does_not_create_document_row(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()

        response = client.post(
            UPLOADS_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "notes.txt"},
        )

        from uuid import UUID

        document_id = UUID(response.json()["data"]["document_id"])
        assert db.get(Document, document_id) is None

    def test_unsupported_extension_is_rejected(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        response = client.post(
            UPLOADS_ROUTE,
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
            UPLOADS_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": ""},
        )

        assert response.status_code == 422

    def test_missing_api_key_is_unauthorized(self, client: TestClient) -> None:
        response = client.post(UPLOADS_ROUTE, json={"filename": "notes.txt"})

        assert response.status_code == 401

    def test_invalid_api_key_is_unauthorized(self, client: TestClient) -> None:
        response = client.post(
            UPLOADS_ROUTE,
            headers={"X-API-KEY": "ApiKey not-a-real-key"},
            json={"filename": "notes.txt"},
        )

        assert response.status_code == 401
