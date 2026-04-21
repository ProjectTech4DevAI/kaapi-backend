"""Tests for load_secret_from_aws helper in app.core.config."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import load_secret_from_aws


@pytest.fixture(autouse=True)
def _clean_aws_env(monkeypatch):
    """Strip any AWS-related env vars that may leak from .env.test."""
    for var in (
        "USE_AWS_SECRETS",
        "AWS_SECRETS_NAMES",
        "AWS_SECRETS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(var, raising=False)


class TestLoadSecretFromAws:
    def test_toggle_off_returns_empty(self, monkeypatch):
        monkeypatch.setenv("USE_AWS_SECRETS", "false")
        assert (
            load_secret_from_aws("any-secret", {"password": "POSTGRES_PASSWORD"}) == {}
        )

    def test_toggle_unset_returns_empty(self):
        assert (
            load_secret_from_aws("any-secret", {"password": "POSTGRES_PASSWORD"}) == {}
        )

    def test_empty_secret_names_returns_empty(self, monkeypatch):
        monkeypatch.setenv("USE_AWS_SECRETS", "true")
        assert load_secret_from_aws("", {"password": "POSTGRES_PASSWORD"}) == {}

    def test_whitespace_only_secret_names_returns_empty(self, monkeypatch):
        monkeypatch.setenv("USE_AWS_SECRETS", "true")
        assert load_secret_from_aws(" , ,", {"password": "POSTGRES_PASSWORD"}) == {}

    @patch("boto3.client")
    def test_single_secret_maps_all_fields(self, mock_boto, monkeypatch):
        monkeypatch.setenv("USE_AWS_SECRETS", "true")
        monkeypatch.setenv("AWS_SECRETS_REGION", "ap-south-1")
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps(
                {"username": "u", "password": "p", "host": "h", "port": 5432}
            )
        }
        mock_boto.return_value = mock_client

        result = load_secret_from_aws(
            "kaapi-db",
            {
                "username": "POSTGRES_USER",
                "password": "POSTGRES_PASSWORD",
                "host": "POSTGRES_SERVER",
                "port": "POSTGRES_PORT",
            },
        )

        assert result == {
            "POSTGRES_USER": "u",
            "POSTGRES_PASSWORD": "p",
            "POSTGRES_SERVER": "h",
            "POSTGRES_PORT": 5432,
        }
        mock_boto.assert_called_once_with("secretsmanager", region_name="ap-south-1")
        mock_client.get_secret_value.assert_called_once_with(SecretId="kaapi-db")

    @patch("boto3.client")
    def test_missing_keys_in_secret_are_skipped(self, mock_boto, monkeypatch):
        monkeypatch.setenv("USE_AWS_SECRETS", "true")
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"password": "p"})
        }
        mock_boto.return_value = mock_client

        result = load_secret_from_aws(
            "kaapi-db",
            {"username": "POSTGRES_USER", "password": "POSTGRES_PASSWORD"},
        )

        assert result == {"POSTGRES_PASSWORD": "p"}

    @patch("boto3.client")
    def test_multiple_secrets_merge_left_to_right(self, mock_boto, monkeypatch):
        monkeypatch.setenv("USE_AWS_SECRETS", "true")
        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = [
            {"SecretString": json.dumps({"password": "first"})},
            {"SecretString": json.dumps({"password": "second"})},
        ]
        mock_boto.return_value = mock_client

        result = load_secret_from_aws("a,b", {"password": "POSTGRES_PASSWORD"})

        assert result == {"POSTGRES_PASSWORD": "second"}
        assert mock_client.get_secret_value.call_count == 2

    @patch("boto3.client")
    def test_non_dict_payload_raises(self, mock_boto, monkeypatch):
        monkeypatch.setenv("USE_AWS_SECRETS", "true")
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps(["not", "a", "dict"])
        }
        mock_boto.return_value = mock_client

        with pytest.raises(ValueError, match="must be a JSON object"):
            load_secret_from_aws("kaapi-db", {"password": "POSTGRES_PASSWORD"})

    @patch("boto3.client")
    def test_region_falls_back_to_aws_default_region(self, mock_boto, monkeypatch):
        monkeypatch.setenv("USE_AWS_SECRETS", "true")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"password": "p"})
        }
        mock_boto.return_value = mock_client

        load_secret_from_aws("kaapi-db", {"password": "POSTGRES_PASSWORD"})

        mock_boto.assert_called_once_with("secretsmanager", region_name="us-east-1")

    @patch("boto3.client")
    def test_no_region_builds_client_without_region_kwarg(self, mock_boto, monkeypatch):
        monkeypatch.setenv("USE_AWS_SECRETS", "true")
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"password": "p"})
        }
        mock_boto.return_value = mock_client

        load_secret_from_aws("kaapi-db", {"password": "POSTGRES_PASSWORD"})

        mock_boto.assert_called_once_with("secretsmanager")
