import pytest

from app.core.providers import (
    PROVIDER_CONFIGS,
    Provider,
    get_supported_providers,
    mask_credential_fields,
    validate_provider,
    validate_provider_credentials,
)


def test_validate_provider_invalid():
    """Test validating an invalid provider name."""
    with pytest.raises(ValueError) as exc_info:
        validate_provider("invalid_provider")
    assert "Unsupported provider" in str(exc_info.value)
    assert "openai" in str(exc_info.value)  # Check that supported providers are listed


def test_validate_provider_credentials_missing_fields():
    """Test validating provider credentials with missing required fields."""
    # Test OpenAI missing api_key
    with pytest.raises(ValueError) as exc_info:
        validate_provider_credentials("openai", {})
    assert "Missing required fields" in str(exc_info.value)
    assert "api_key" in str(exc_info.value)
