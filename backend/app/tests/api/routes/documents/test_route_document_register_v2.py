from unittest.mock import patch
from urllib.parse import quote
from uuid import UUID, uuid4

import pytest
import requests
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from httpx import Response
from moto import mock_aws
from sqlmodel import Session

from app.core.cloud import AmazonCloudStorageClient
from app.core.config import settings
from app.core.db import engine
from app.core.util import now
from app.crud import DocumentCrud
from app.models import Document
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.document import DocumentMaker

DOCUMENTS_ROUTE = f"{settings.API_V2_STR}/documents"
UPLOADS_ROUTE = f"{DOCUMENTS_ROUTE}/uploads"


def pending_key(auth: TestAuthContext, document_id: UUID) -> str:
    # No extension: the filename rides in signed metadata, not in the key.
    return f"pending/{auth.project.storage_path}/{document_id}"


def final_key(auth: TestAuthContext, document_id: UUID) -> str:
    return f"{auth.project.storage_path}/{document_id}"


def put_pending(
    auth: TestAuthContext, document_id: UUID, body: bytes, filename: str
) -> None:
    """Simulate a client upload: the object plus the filename the ticket would have pinned."""
    AmazonCloudStorageClient().client.put_object(
        Bucket=settings.AWS_S3_BUCKET,
        Key=pending_key(auth, document_id),
        Body=body,
        Metadata={"filename": quote(filename)},
    )


def assert_absent(key: str) -> None:
    with pytest.raises(ClientError) as excinfo:
        AmazonCloudStorageClient().client.head_object(
            Bucket=settings.AWS_S3_BUCKET, Key=key
        )
    assert excinfo.value.response["Error"]["Code"] == "404"


def register(client: TestClient, auth: TestAuthContext, document_id: UUID) -> Response:
    return client.put(
        f"{DOCUMENTS_ROUTE}/{document_id}",
        headers={"X-API-KEY": auth.key},
    )


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestDocumentRegisterV2:
    def test_registers_pending_object_under_its_final_key(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()
        document_id = uuid4()
        put_pending(user_api_key, document_id, b"x" * 2048, "report.pdf")

        response = register(client, user_api_key, document_id)

        assert response.status_code == 201
        data = response.json()["data"]
        assert data["id"] == str(document_id)
        assert data["fname"] == "report.pdf"
        assert data["signed_url"]
        assert "cannot be reused" in response.json()["metadata"]["note"]

        document = db.get(Document, document_id)
        assert document is not None
        assert document.fname == "report.pdf"
        assert document.file_size_kb == 2.0
        assert document.object_store_url == (
            f"s3://{settings.AWS_S3_BUCKET}/{final_key(user_api_key, document_id)}"
        )
        assert document.project_id == user_api_key.project_id

        assert_absent(pending_key(user_api_key, document_id))

    def test_missing_object_is_rejected(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()
        document_id = uuid4()

        response = register(client, user_api_key, document_id)

        assert response.status_code == 400
        assert "No uploaded file found" in response.json()["error"]
        assert db.get(Document, document_id) is None

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
        put_pending(user_api_key, existing.id, b"x" * 1024, "report.pdf")

        response = register(client, user_api_key, existing.id)

        assert response.status_code == 409
        assert "already registered" in response.json()["error"]

        db.refresh(existing)
        assert existing.fname != "report.pdf"

    def test_concurrent_registration_loses_the_insert_race(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()
        winner = next(DocumentMaker(project_id=user_api_key.project_id, session=db))
        document_id, original_fname = winner.id, winner.fname
        # Committed outside the test transaction: the winner of the race is another
        # request's session, so the request under test can only hit the PK constraint.
        with Session(engine) as outside:
            outside.add(winner)
            outside.commit()
        put_pending(user_api_key, document_id, b"x" * 1024, "report.pdf")

        try:
            # exists() returning False simulates the racing request that also saw no row.
            with patch.object(DocumentCrud, "exists", return_value=False):
                response = register(client, user_api_key, document_id)

            assert response.status_code == 409
            assert "already registered" in response.json()["error"]

            survivor = db.get(Document, document_id)
            assert survivor is not None
            assert survivor.fname == original_fname
        finally:
            with Session(engine) as cleanup:
                row = cleanup.get(Document, document_id)
                if row is not None:
                    cleanup.delete(row)
                    cleanup.commit()

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

        response = register(client, user_api_key, deleted.id)

        assert response.status_code == 409
        assert "already registered" in response.json()["error"]

    def test_missing_api_key_is_unauthorized(self, client: TestClient) -> None:
        response = client.put(f"{DOCUMENTS_ROUTE}/{uuid4()}")

        assert response.status_code == 401


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestDocumentUploadRoundTripV2:
    def test_uploads_then_post_then_register(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        AmazonCloudStorageClient().create()

        init = client.post(
            UPLOADS_ROUTE,
            headers={"X-API-KEY": user_api_key.key},
            json={"filename": "handbook.pdf"},
        )
        assert init.status_code == 200
        data = init.json()["data"]
        document_id = UUID(data["document_id"])

        upload = requests.post(
            data["upload_url"],
            data=data["upload_fields"],
            files={"file": ("handbook.pdf", b"y" * 3072)},
        )
        assert upload.status_code in (200, 204)

        response = register(client, user_api_key, document_id)

        assert response.status_code == 201
        assert response.json()["data"]["id"] == str(document_id)

        document = db.get(Document, document_id)
        assert document is not None
        assert document.object_store_url == (
            f"s3://{settings.AWS_S3_BUCKET}/{final_key(user_api_key, document_id)}"
        )
        assert_absent(pending_key(user_api_key, document_id))
