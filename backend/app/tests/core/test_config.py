"""Tests for the _apply_aws_secrets validator on Settings."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings

_REQUIRED_FIELDS = {
    "PROJECT_NAME": "test",
    "POSTGRES_SERVER": "localhost",
    "POSTGRES_USER": "env-user",
    "POSTGRES_PASSWORD": "env-pw",
    "EMAIL_TEST_USER": "t@example.com",
    "FIRST_SUPERUSER": "s@example.com",
    "FIRST_SUPERUSER_PASSWORD": "superpw",
}


@pytest.fixture(autouse=True)
def _clean_aws_env(monkeypatch):
    """Strip any AWS-related env vars that may leak from .env.test."""
    for var in (
        "USE_AWS_SECRETS",
        "AWS_POSTGRES_SECRET_NAME",
        "AWS_SECRETS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_settings(**overrides) -> Settings:
    """Build a Settings instance bypassing the .env file."""
    return Settings(_env_file=None, **{**_REQUIRED_FIELDS, **overrides})


class TestApplyAwsSecrets:
    def test_toggle_off_keeps_env_values(self):
        s = _make_settings(USE_AWS_SECRETS=False)
        assert s.POSTGRES_PASSWORD == "env-pw"
        assert s.POSTGRES_USER == "env-user"

    def test_toggle_on_but_no_secret_names_keeps_env_values(self):
        s = _make_settings(USE_AWS_SECRETS=True)
        assert s.POSTGRES_PASSWORD == "env-pw"
        assert s.POSTGRES_USER == "env-user"

    @patch("boto3.client")
    def test_postgres_secret_overrides_postgres_fields(self, mock_boto):
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps(
                {"username": "u", "password": "p", "host": "h", "port": 5433}
            )
        }
        mock_boto.return_value = mock_client

        setting = _make_settings(
            USE_AWS_SECRETS=True,
            AWS_POSTGRES_SECRET_NAME="kaapi-db",
            AWS_SECRETS_REGION="ap-south-1",
        )

        assert setting.POSTGRES_USER == "u"
        assert setting.POSTGRES_PASSWORD == "p"
        assert setting.POSTGRES_SERVER == "h"
        assert setting.POSTGRES_PORT == 5433
        mock_boto.assert_called_once_with("secretsmanager", region_name="ap-south-1")
        mock_client.get_secret_value.assert_called_once_with(SecretId="kaapi-db")

    @patch("boto3.client")
    def test_missing_keys_fall_back_to_env(self, mock_boto):
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"password": "aws-pw"})
        }
        mock_boto.return_value = mock_client

        setting = _make_settings(
            USE_AWS_SECRETS=True,
            AWS_POSTGRES_SECRET_NAME="kaapi-db",
        )

        assert setting.POSTGRES_PASSWORD == "aws-pw"
        assert setting.POSTGRES_USER == "env-user"
        assert setting.POSTGRES_SERVER == "localhost"

    @patch("boto3.client")
    def test_non_dict_payload_raises(self, mock_boto):
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps(["not", "a", "dict"])
        }
        mock_boto.return_value = mock_client

        with pytest.raises(ValueError, match="must be a JSON object"):
            _make_settings(
                USE_AWS_SECRETS=True,
                AWS_POSTGRES_SECRET_NAME="kaapi-db",
            )

    @patch("boto3.client")
    def test_region_falls_back_to_aws_default_region(self, mock_boto):
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"password": "p"})
        }
        mock_boto.return_value = mock_client

        _make_settings(
            USE_AWS_SECRETS=True,
            AWS_POSTGRES_SECRET_NAME="kaapi-db",
            AWS_DEFAULT_REGION="us-east-1",
        )

        mock_boto.assert_called_once_with("secretsmanager", region_name="us-east-1")

    @patch("boto3.client")
    def test_no_region_builds_client_without_kwarg(self, mock_boto):
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"password": "p"})
        }
        mock_boto.return_value = mock_client

        _make_settings(
            USE_AWS_SECRETS=True,
            AWS_POSTGRES_SECRET_NAME="kaapi-db",
        )

        mock_boto.assert_called_once_with("secretsmanager")
