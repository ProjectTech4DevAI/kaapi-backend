"""Tests for app.core.cloud.storage helpers."""

import os
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from app.core.cloud.storage import (
    GCS_SCOPES,
    AmazonCloudStorage,
    CloudStorageError,
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


@pytest.fixture(scope="class")
def aws_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION


@mock_aws
@pytest.mark.usefixtures("aws_credentials")
class TestGetSignedUploadURL:
    def test_url_targets_the_requested_key(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())
        key = f"{storage.storage_path}/{uuid4()}"

        url = storage.get_signed_upload_url(key)

        parsed = urlparse(url)
        assert parsed.path.endswith(key)
        assert settings.AWS_S3_BUCKET in f"{parsed.netloc}{parsed.path}"
        assert "X-Amz-Signature" in parse_qs(parsed.query)

    def test_content_type_is_signed_when_given(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        url = storage.get_signed_upload_url("key.pdf", content_type="application/pdf")

        signed_headers = parse_qs(urlparse(url).query)["X-Amz-SignedHeaders"][0]
        assert "content-type" in signed_headers

    def test_expiry_is_capped_at_one_day(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        url = storage.get_signed_upload_url("key.pdf", expires_in=7 * 24 * 3600)

        assert parse_qs(urlparse(url).query)["X-Amz-Expires"] == ["86400"]

    def test_shorter_expiry_is_preserved(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())

        url = storage.get_signed_upload_url("key.pdf", expires_in=600)

        assert parse_qs(urlparse(url).query)["X-Amz-Expires"] == ["600"]

    def test_aws_error_is_wrapped(self) -> None:
        storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())
        error = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}},
            "PutObject",
        )

        with patch.object(
            storage.aws.client, "generate_presigned_url", side_effect=error
        ):
            with pytest.raises(CloudStorageError, match="AccessDenied"):
                storage.get_signed_upload_url("key.pdf")
