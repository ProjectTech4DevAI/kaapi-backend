"""Tests for app.core.cloud.storage helpers."""

from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from app.core.cloud.storage import (
    GCS_SCOPES,
    AmazonCloudStorage,
    AmazonCloudStorageClient,
    CloudStorageError,
    ObjectNotFoundError,
    SimpleStorageName,
    _to_storage_error,
    build_gcp_sa_credentials,
)
from app.core.config import settings


def test_build_gcp_sa_credentials_passes_key_and_scopes():
    sa_key = {"type": "service_account", "project_id": "p"}
    with patch(
        "app.core.cloud.storage.service_account.Credentials.from_service_account_info"
    ) as mock_from_info:
        creds = build_gcp_sa_credentials(sa_key)

    mock_from_info.assert_called_once_with(sa_key, scopes=list(GCS_SCOPES))
    assert creds is mock_from_info.return_value


def client_error(code: str, operation: str = "HeadObject") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class TestUrlFor:
    def test_joins_the_projects_storage_path(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        name = storage.url_for(Path("report.pdf"))

        assert name.Key == f"{storage.storage_path}/report.pdf"
        assert name.Bucket == settings.AWS_S3_BUCKET

    def test_pending_prefix_leads_the_key(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        name = storage.url_for(Path("report.pdf"), is_pending=True)

        # Ahead of storage_path, not after it: one literal-prefix rule must match every project.
        assert name.Key == f"pending/{storage.storage_path}/report.pdf"

    def test_absolute_path_is_rejected(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        with pytest.raises(ValueError, match="must be relative"):
            storage.url_for(Path("/etc/passwd"))


class TestToStorageError:
    @pytest.mark.parametrize("code", ["404", "NoSuchKey"])
    def test_missing_object_codes_become_object_not_found(self, code: str) -> None:
        error = _to_storage_error(client_error(code), "s3://bucket/key")

        assert isinstance(error, ObjectNotFoundError)
        assert "s3://bucket/key" in str(error)

    def test_other_codes_stay_generic(self) -> None:
        error = _to_storage_error(client_error("AccessDenied"), "s3://bucket/key")

        assert isinstance(error, CloudStorageError)
        assert not isinstance(error, ObjectNotFoundError)


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestCreateUploadTicket:
    def test_targets_the_pending_key(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())
        document_id = uuid4()

        ticket = storage.create_upload_ticket(
            Path(str(document_id)), filename="report.pdf", max_bytes=25 * 1024 * 1024
        )

        assert ticket.fields["key"] == f"pending/{storage.storage_path}/{document_id}"
        assert "x-amz-signature" in ticket.fields

    def test_filename_is_pinned_url_encoded(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        ticket = storage.create_upload_ticket(
            Path(str(uuid4())), filename="my report.pdf", max_bytes=1024
        )

        assert ticket.fields["x-amz-meta-filename"] == quote("my report.pdf")

    def test_expiry_is_capped_at_one_day(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        ticket = storage.create_upload_ticket(
            Path("f"), filename="a.pdf", max_bytes=1024, expires_in=7 * 24 * 3600
        )

        assert ticket.expires_in == 86400

    def test_shorter_expiry_is_preserved(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        ticket = storage.create_upload_ticket(
            Path("f"), filename="a.pdf", max_bytes=1024, expires_in=600
        )

        assert ticket.expires_in == 600

    def test_aws_error_is_wrapped(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        with patch.object(
            storage.aws.client,
            "generate_presigned_post",
            side_effect=client_error("AccessDenied", "PutObject"),
        ):
            with pytest.raises(CloudStorageError, match="AccessDenied"):
                storage.create_upload_ticket(
                    Path("f"), filename="a.pdf", max_bytes=1024
                )


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestHead:
    def test_returns_size_and_decoded_filename(self) -> None:
        aws = AmazonCloudStorageClient()
        aws.create()
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())
        name = storage.url_for(Path(str(uuid4())), is_pending=True)
        aws.client.put_object(
            Bucket=name.Bucket,
            Key=name.Key,
            Body=b"x" * 2048,
            Metadata={"filename": quote("my report.pdf")},
        )

        stored = storage.head(str(name))

        assert stored.size_kb == 2.0
        assert stored.filename == "my report.pdf"

    def test_filename_is_none_without_metadata(self) -> None:
        aws = AmazonCloudStorageClient()
        aws.create()
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())
        name = storage.url_for(Path(str(uuid4())))
        aws.client.put_object(Bucket=name.Bucket, Key=name.Key, Body=b"x" * 1024)

        stored = storage.head(str(name))

        assert stored.size_kb == 1.0
        assert stored.filename is None

    def test_missing_key_raises_object_not_found(self) -> None:
        AmazonCloudStorageClient().create()
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())
        url = str(storage.url_for(Path(str(uuid4()))))

        with pytest.raises(ObjectNotFoundError):
            storage.head(url)


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestCopy:
    def test_copies_the_source_bytes_to_the_destination_key(self) -> None:
        aws = AmazonCloudStorageClient()
        aws.create()
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())
        source = storage.url_for(Path("waiting.pdf"), is_pending=True)
        aws.client.put_object(
            Bucket=source.Bucket, Key=source.Key, Body=b"pending bytes"
        )
        destination = Path("final.pdf")

        target = storage.copy(str(source), destination)

        assert target == SimpleStorageName(
            Key=f"{storage.storage_path}/final.pdf", Bucket=settings.AWS_S3_BUCKET
        )
        copied = aws.client.get_object(Bucket=target.Bucket, Key=target.Key)
        assert copied["Body"].read() == b"pending bytes"

    def test_missing_source_raises_object_not_found(self) -> None:
        AmazonCloudStorageClient().create()
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())
        source = str(storage.url_for(Path("absent.pdf"), is_pending=True))

        with pytest.raises(ObjectNotFoundError):
            storage.copy(source, Path("final.pdf"))
