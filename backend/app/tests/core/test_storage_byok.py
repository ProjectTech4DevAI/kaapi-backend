"""Tests for the BYOK helpers in app.core.cloud.storage."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.core.cloud.storage import (
    SecretsManagerError,
    put_gcp_service_account,
    upsert_byok_secret_for_provider,
)


VALID_SA = {
    "type": "service_account",
    "project_id": "starlit-lotus-492004-k0",
    "client_email": "kaapi-test@starlit-lotus-492004-k0.iam.gserviceaccount.com",
    "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----",
}


@pytest.fixture
def mock_sm_client():
    client = MagicMock()
    client.exceptions.ResourceExistsException = type(
        "ResourceExistsException", (ClientError,), {}
    )
    with patch("app.core.cloud.storage.boto3.session.Session") as mock_session, patch(
        "app.core.cloud.storage.get_gcp_service_account.cache_clear"
    ) as mock_clear:
        mock_session.return_value.client.return_value = client
        yield client, mock_clear


class TestPutGcpServiceAccount:
    def test_creates_secret_when_absent(self, mock_sm_client):
        client, mock_clear = mock_sm_client
        put_gcp_service_account(
            VALID_SA, secret_name="kaapi/dev/orgs/1/projects/2/google-vertex/sa"
        )

        client.create_secret.assert_called_once()
        kwargs = client.create_secret.call_args.kwargs
        assert kwargs["Name"] == "kaapi/dev/orgs/1/projects/2/google-vertex/sa"
        # SA JSON round-trips through json.dumps; verify a known field survives.
        assert '"type": "service_account"' in kwargs["SecretString"]
        client.put_secret_value.assert_not_called()
        mock_clear.assert_called_once()

    def test_updates_secret_when_present(self, mock_sm_client):
        client, mock_clear = mock_sm_client
        client.create_secret.side_effect = client.exceptions.ResourceExistsException(
            {"Error": {"Code": "ResourceExistsException"}}, "CreateSecret"
        )

        put_gcp_service_account(
            VALID_SA, secret_name="kaapi/dev/orgs/1/projects/2/google-vertex/sa"
        )

        client.create_secret.assert_called_once()
        client.put_secret_value.assert_called_once()
        kwargs = client.put_secret_value.call_args.kwargs
        assert kwargs["SecretId"] == "kaapi/dev/orgs/1/projects/2/google-vertex/sa"
        mock_clear.assert_called_once()

    def test_rejects_non_service_account_payload(self, mock_sm_client):
        client, _ = mock_sm_client
        bad = {"type": "user_account", "client_id": "x"}
        with pytest.raises(SecretsManagerError, match="not a GCP service-account"):
            put_gcp_service_account(bad, secret_name="kaapi/anything")
        client.create_secret.assert_not_called()
        client.put_secret_value.assert_not_called()

    def test_wraps_aws_errors(self, mock_sm_client):
        client, _ = mock_sm_client
        client.create_secret.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "nope"}},
            "CreateSecret",
        )
        with pytest.raises(SecretsManagerError, match="AccessDeniedException"):
            put_gcp_service_account(VALID_SA, secret_name="kaapi/anything")


class TestUpsertByokSecretForProvider:
    def test_google_vertex_with_sa_key_strips_and_writes(self):
        creds = {
            "api_key": "vkey",
            "project_id": "starlit-lotus-492004-k0",
            "location": "us-central1",
            "sa_key": VALID_SA,
            "gcs_bucket": "my-bucket",
        }
        with patch("app.core.cloud.storage.put_gcp_service_account") as mock_put, patch(
            "app.core.cloud.storage.settings"
        ) as mock_settings:
            mock_settings.ENVIRONMENT = "development"
            mock_settings.GCP_SA_SECRET_REGION = "ap-south-1"

            result = upsert_byok_secret_for_provider(
                "google-vertex", creds, org_id=7, project_id=42
            )

        expected_name = "kaapi/development/orgs/7/projects/42/google-vertex/sa"
        mock_put.assert_called_once_with(VALID_SA, secret_name=expected_name)
        assert "sa_key" not in result
        assert result["gcp_sa_secret_name"] == expected_name
        assert result["gcp_sa_secret_region"] == "ap-south-1"
        # Untouched fields preserved.
        assert result["api_key"] == "vkey"
        assert result["gcs_bucket"] == "my-bucket"

    def test_google_vertex_rejects_null_sa_key(self):
        creds = {
            "api_key": "vkey",
            "project_id": "p",
            "location": "us-central1",
            "sa_key": None,
        }
        with pytest.raises(ValueError, match="sa_key.*non-empty service-account JSON"):
            upsert_byok_secret_for_provider(
                "google-vertex", creds, org_id=1, project_id=1
            )

    def test_google_vertex_rejects_string_sa_key(self):
        creds = {
            "api_key": "vkey",
            "project_id": "p",
            "location": "us-central1",
            "sa_key": "not-a-dict",
        }
        with pytest.raises(ValueError, match="non-empty service-account JSON"):
            upsert_byok_secret_for_provider(
                "google-vertex", creds, org_id=1, project_id=1
            )

    def test_google_vertex_rejects_empty_sa_key(self):
        creds = {
            "api_key": "vkey",
            "project_id": "p",
            "location": "us-central1",
            "sa_key": {},
        }
        with pytest.raises(ValueError, match="non-empty service-account JSON"):
            upsert_byok_secret_for_provider(
                "google-vertex", creds, org_id=1, project_id=1
            )

    def test_google_vertex_rejects_missing_sa_key(self):
        # Validator at the route requires sa_key, but the hook also rejects
        # absence defensively in case it's invoked outside the route flow.
        creds = {"api_key": "vkey", "project_id": "p", "location": "us-central1"}
        with pytest.raises(ValueError, match="non-empty service-account JSON"):
            upsert_byok_secret_for_provider(
                "google-vertex", creds, org_id=1, project_id=1
            )

    def test_other_provider_is_noop_even_with_sa_key(self):
        creds = {"api_key": "k", "sa_key": VALID_SA}
        with patch("app.core.cloud.storage.put_gcp_service_account") as mock_put:
            result = upsert_byok_secret_for_provider(
                "openai", creds, org_id=1, project_id=1
            )
        mock_put.assert_not_called()
        assert result == creds  # sa_key passes through (validator's job to reject)
