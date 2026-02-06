import logging
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Enumeration of supported credential providers."""

    OPENAI = "openai"
    AWS = "aws"
    LANGFUSE = "langfuse"
    GOOGLE = "google"


@dataclass
class ProviderConfig:
    """Configuration for a provider including its required credential fields."""

    required_fields: List[str]


# Provider configurations
PROVIDER_CONFIGS: Dict[Provider, ProviderConfig] = {
    Provider.OPENAI: ProviderConfig(required_fields=["api_key"]),
    Provider.AWS: ProviderConfig(
        required_fields=["access_key_id", "secret_access_key", "region"]
    ),
    Provider.LANGFUSE: ProviderConfig(
        required_fields=["secret_key", "public_key", "host"]
    ),
    Provider.GOOGLE: ProviderConfig(required_fields=["api_key"]),
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
        logger.error(
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
        logger.error(
            f"[validate_provider_credentials] Missing required fields | provider: {provider}, missing_fields: {', '.join(missing_fields)}"
        )
        raise ValueError(
            f"Missing required fields for {provider}: {', '.join(missing_fields)}"
        )


def get_supported_providers() -> List[str]:
    """Return a list of all supported provider names."""
    return [p.value for p in Provider]
