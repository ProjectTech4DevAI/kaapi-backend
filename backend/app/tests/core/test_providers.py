import json

import pytest

from app.core.providers import parse_provider_credentials, validate_provider
from app.models.provider_credentials import GoogleGcpCredentials

_SA_KEY = {"type": "service_account", "project_id": "proj"}
_GCP_CRED = {
    "api_key": "test-key",
    "project_id": "proj",
    "location": "us-central1",
    "gcs_bucket": "bucket",
}


def test_validate_provider_invalid():
    """Test validating an invalid provider name."""
    with pytest.raises(ValueError) as exc_info:
        validate_provider("invalid_provider")
    assert "Unsupported provider" in str(exc_info.value)
    assert "openai" in str(exc_info.value)  # Check that supported providers are listed


def test_parse_provider_credentials_missing_fields():
    """Test validating provider credentials with missing required fields."""
    # Test OpenAI missing api_key
    with pytest.raises(ValueError) as exc_info:
        parse_provider_credentials("openai", {})
    assert "Missing required fields" in str(exc_info.value)
    assert "api_key" in str(exc_info.value)


def test_parse_gcp_credentials_accepts_sa_key_object():
    creds = parse_provider_credentials("google-gcp", {**_GCP_CRED, "sa_key": _SA_KEY})
    assert isinstance(creds, GoogleGcpCredentials)
    assert creds.sa_key == _SA_KEY


@pytest.mark.parametrize(
    "sa_key", [json.dumps(_SA_KEY), "not-json", "[1, 2]", 42, ["x"]]
)
def test_parse_gcp_credentials_rejects_non_object_sa_key(sa_key: object):
    with pytest.raises(ValueError, match="sa_key"):
        parse_provider_credentials("google-gcp", {**_GCP_CRED, "sa_key": sa_key})
