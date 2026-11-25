import base64
import json
from urllib.parse import quote
from uuid import UUID, uuid4

import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException
from moto import mock_aws
from sqlmodel import Session

from app.core.cloud import AmazonCloudStorageClient
from app.core.config import settings
from app.models import Document, DocumentPublic
from app.services.documents import (
    initiate_document_upload,
    register_uploaded_document,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.document import DocumentMaker

# 25 MiB, written out rather than imported so a change to the constant is caught.
EXPECTED_MAX_UPLOAD_BYTES = 26214400
EXPECTED_EXPIRY_SECONDS = 3600


def pending_key(auth: TestAuthContext, document_id: UUID) -> str:
    return f"pending/{auth.project.storage_path}/{document_id}"


def put_pending(
    auth: TestAuthContext, document_id: UUID, body: bytes, filename: str
) -> None:
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


def decode_policy(fields: dict[str, str]) -> dict:
    return json.loads(base64.b64decode(fields["policy"]))


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestInitiateDocumentUpload:
    def test_returns_ticket_without_persisting_a_row(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        AmazonCloudStorageClient().create()

        result = initiate_document_upload(
            session=db, project_id=user_api_key.project_id, filename="notes.txt"
        )

        assert result.upload_url
        assert result.upload_fields["key"] == pending_key(
            user_api_key, result.document_id
        )
        assert db.get(Document, result.document_id) is None

    def test_ticket_carries_size_cap_and_expiry(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        AmazonCloudStorageClient().create()

        result = initiate_document_upload(
            session=db, project_id=user_api_key.project_id, filename="notes.txt"
        )

        assert result.expires_in == EXPECTED_EXPIRY_SECONDS
        conditions = decode_policy(result.upload_fields)["conditions"]
        assert ["content-length-range", 1, EXPECTED_MAX_UPLOAD_BYTES] in conditions

    def test_unsupported_extension_is_rejected(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        with pytest.raises(HTTPException) as excinfo:
            initiate_document_upload(
                session=db, project_id=user_api_key.project_id, filename="notes.xyz"
            )

        assert excinfo.value.status_code == 400
        assert "Unsupported file extension: .xyz" in excinfo.value.detail


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestRegisterUploadedDocument:
    def test_returns_public_document_and_clears_pending_object(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        AmazonCloudStorageClient().create()
        document_id = uuid4()
        put_pending(user_api_key, document_id, b"x" * 2048, "report.pdf")

        result = register_uploaded_document(
            session=db, project_id=user_api_key.project_id, document_id=document_id
        )

        assert isinstance(result, DocumentPublic)
        assert result.id == document_id
        assert result.fname == "report.pdf"
        assert result.signed_url
        assert db.get(Document, document_id) is not None
        assert_absent(pending_key(user_api_key, document_id))

    def test_duplicate_document_id_is_rejected(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        AmazonCloudStorageClient().create()
        existing = next(DocumentMaker(project_id=user_api_key.project_id, session=db))
        db.add(existing)
        db.commit()
        put_pending(user_api_key, existing.id, b"x" * 1024, "report.pdf")

        with pytest.raises(HTTPException) as excinfo:
            register_uploaded_document(
                session=db, project_id=user_api_key.project_id, document_id=existing.id
            )

        assert excinfo.value.status_code == 409
        db.refresh(existing)
        assert existing.fname != "report.pdf"

    def test_missing_upload_is_rejected(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        AmazonCloudStorageClient().create()
        document_id = uuid4()

        with pytest.raises(HTTPException) as excinfo:
            register_uploaded_document(
                session=db, project_id=user_api_key.project_id, document_id=document_id
            )

        assert excinfo.value.status_code == 400
        assert "No uploaded file found" in excinfo.value.detail
        assert db.get(Document, document_id) is None
