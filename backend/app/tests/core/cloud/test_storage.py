"""Tests for app.core.cloud.storage helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.core.cloud.storage import (
    GCS_SCOPES,
    AmazonCloudStorage,
    build_gcp_sa_credentials,
)


def test_build_gcp_sa_credentials_passes_key_and_scopes():
    sa_key = {"type": "service_account", "project_id": "p"}
    with patch(
        "app.core.cloud.storage.service_account.Credentials.from_service_account_info"
    ) as mock_from_info:
        creds = build_gcp_sa_credentials(sa_key)

    mock_from_info.assert_called_once_with(sa_key, scopes=list(GCS_SCOPES))
    assert creds is mock_from_info.return_value


def _amazon_storage_with_mock_client(mock_client):
    storage = AmazonCloudStorage(project_id=1, storage_path=uuid4())
    storage.aws = SimpleNamespace(client=mock_client)
    return storage


def test_get_signed_url_without_filename_omits_content_disposition():
    mock_client = MagicMock()
    storage = _amazon_storage_with_mock_client(mock_client)

    storage.get_signed_url("s3://bucket/key.pdf")

    params = mock_client.generate_presigned_url.call_args.kwargs["Params"]
    assert "ResponseContentDisposition" not in params


def test_get_signed_url_with_filename_forces_attachment():
    mock_client = MagicMock()
    storage = _amazon_storage_with_mock_client(mock_client)

    storage.get_signed_url("s3://bucket/key.pdf", filename="report.pdf")

    disposition = mock_client.generate_presigned_url.call_args.kwargs["Params"][
        "ResponseContentDisposition"
    ]
    assert (
        disposition
        == "attachment; filename=\"report.pdf\"; filename*=UTF-8''report.pdf"
    )


def test_get_signed_url_encodes_non_ascii_filename():
    mock_client = MagicMock()
    storage = _amazon_storage_with_mock_client(mock_client)

    storage.get_signed_url("s3://bucket/key.pdf", filename="रिपोर्ट.pdf")

    disposition = mock_client.generate_presigned_url.call_args.kwargs["Params"][
        "ResponseContentDisposition"
    ]
    assert disposition.startswith('attachment; filename="')
    assert "filename*=UTF-8''%E0%A4%B0" in disposition
