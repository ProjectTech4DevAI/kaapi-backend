"""Tests for app.core.cloud.storage helpers."""

from unittest.mock import patch

from app.core.cloud.storage import GCS_SCOPES, build_gcp_sa_credentials


def test_build_gcp_sa_credentials_passes_key_and_scopes():
    sa_key = {"type": "service_account", "project_id": "p"}
    with patch(
        "app.core.cloud.storage.service_account.Credentials.from_service_account_info"
    ) as mock_from_info:
        creds = build_gcp_sa_credentials(sa_key)

    mock_from_info.assert_called_once_with(sa_key, scopes=list(GCS_SCOPES))
    assert creds is mock_from_info.return_value
