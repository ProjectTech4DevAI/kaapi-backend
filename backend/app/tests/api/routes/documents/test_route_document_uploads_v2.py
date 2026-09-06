from urllib.parse import urlparse
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlmodel import Session

from app.core.cloud import AmazonCloudStorageClient
from app.core.config import settings
from app.models import Document
from app.tests.utils.auth import TestAuthContext

UPLOADS_ROUTE = f"{settings.API_V2_STR}/documents/uploads"


def signed_key(upload_url: str) -> str:
    """The object key a pre-signed URL points at, with host and bucket stripped."""
    path = urlparse(upload_url).path.lstrip("/")
    return path.removeprefix(f"{settings.AWS_S3_BUCKET}/")


def pending_key(auth: TestAuthContext, document_id: str, extension: str) -> str:
    # The pending prefix leads the key so one S3 lifecycle rule covers every project.
    return f"pending/{auth.project.storage_path}/{document_id}{extension}"


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestDocumentUploadsV2:
    def test_returns_presigned_put_url_for_pending_key(
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

        document_id = UUID(data["document_id"])
        assert (
            pending_key(user_api_key, str(document_id), ".pdf")
            in data["upload_signed_url"]
        )
        assert "X-Amz-Signature" in data["upload_signed_url"]

    def test_upload_url_does_not_target_the_final_key(
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
        # Compared whole: the final key is a substring of the pending one, so `not in` never holds.
        key = signed_key(data["upload_signed_url"])
        assert key == pending_key(user_api_key, data["document_id"], ".pdf")
        assert key != f"{user_api_key.project.storage_path}/{data['document_id']}"

    def test_extension_is_lowercased_in_the_pending_key(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()

        response = client.post(
            UPLOADS_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "Quarterly-Report.PDF"},
        )

        data = response.json()["data"]
        assert (
            pending_key(user_api_key, data["document_id"], ".pdf")
            in data["upload_signed_url"]
        )

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
