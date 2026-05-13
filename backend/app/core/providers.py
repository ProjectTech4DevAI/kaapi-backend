import logging
from typing import Any, Dict, List
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Enumeration of supported credential providers."""

    OPENAI = "openai"
    LANGFUSE = "langfuse"
    GOOGLE = "google"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"
    WEBHOOK_SECRET = "webhook_secret"


@dataclass
class ProviderConfig:
    """Configuration for a provider including its required credential fields."""

    required_fields: List[str]
    sensitive_fields: List[str] = field(default_factory=list)


# Provider configurations
PROVIDER_CONFIGS: Dict[Provider, ProviderConfig] = {
    Provider.OPENAI: ProviderConfig(
        required_fields=["api_key"], sensitive_fields=["api_key"]
    ),
    Provider.LANGFUSE: ProviderConfig(
        required_fields=["secret_key", "public_key", "host"],
        sensitive_fields=["secret_key"],
    ),
    Provider.GOOGLE: ProviderConfig(
        required_fields=["api_key"], sensitive_fields=["api_key"]
    ),
    Provider.SARVAMAI: ProviderConfig(
        required_fields=["api_key"], sensitive_fields=["api_key"]
    ),
    Provider.ELEVENLABS: ProviderConfig(
        required_fields=["api_key"], sensitive_fields=["api_key"]
    ),
    Provider.WEBHOOK_SECRET: ProviderConfig(
        required_fields=["webhook_secret"], sensitive_fields=["webhook_secret"]
    ),
}


def validate_provider(provider: str) -> Provider:
    """Validate that the provider name is supported and return the Provider enum.

    Args:
        provider: The provider name to validate

    Returns:
        Provider: The validated provider enum

    Raises:
        ValueError: If the provider is not supported
    """
    try:
        return Provider(provider.lower())
    except ValueError:
        supported = ", ".join(p.value for p in Provider)
        logger.warning(
            f"[validate_provider] Unsupported provider | provider: {provider}, supported_providers: {supported}"
        )
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers are: {supported}"
        )


def validate_provider_credentials(provider: str, credentials: Dict[str, str]) -> None:
    """Validate that the credentials contain all required fields for the provider.

    Args:
        provider: The provider name to validate credentials for
        credentials: Dictionary containing the provider credentials

    Raises:
        ValueError: If required fields are missing from the credentials
    """
    provider_enum = validate_provider(provider)
    required_fields = PROVIDER_CONFIGS[provider_enum].required_fields

    if missing_fields := [
        field for field in required_fields if field not in credentials
    ]:
        logger.warning(
            f"[validate_provider_credentials] Missing required fields | provider: {provider}, missing_fields: {', '.join(missing_fields)}"
        )
        raise ValueError(
            f"Missing required fields for {provider}: {', '.join(missing_fields)}"
        )


def get_supported_providers() -> List[str]:
    """Return a list of all supported provider names."""
    return [p.value for p in Provider]


def mask_credential_fields(
    provider: str, credentials: Dict[str, Any]
) -> Dict[str, Any]:
    """Mask sensitive fields in a credential dict for the given provider.

    Non-sensitive fields (e.g., langfuse `public_key`, `host`) are returned as-is.
    Unknown providers are returned with no masking.
    """
    from app.utils import mask_string

    if not credentials:
        return credentials

    try:
        provider_enum = Provider(provider.lower())
    except ValueError:
        return credentials

    sensitive_fields = PROVIDER_CONFIGS[provider_enum].sensitive_fields
    masked = dict(credentials)
    for field_name in sensitive_fields:
        if field_name in masked and isinstance(masked[field_name], str):
            masked[field_name] = mask_string(masked[field_name])
    return masked
