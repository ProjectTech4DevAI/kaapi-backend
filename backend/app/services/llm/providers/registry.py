import logging
from sqlmodel import Session

from app.services.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


def _build_registry() -> dict[str, type[BaseProvider]]:
    from app.services.llm.providers.oai import OpenAIProvider
    from app.services.llm.providers.gai import GoogleAIProvider
    from app.services.llm.providers.sai import SarvamAIProvider

    return {
        "openai-native": OpenAIProvider,
        "openai": OpenAIProvider,
        "google": GoogleAIProvider,
        "google-native": GoogleAIProvider,
        "sarvamai-native": SarvamAIProvider,
    }


class LLMProvider:
    OPENAI_NATIVE = "openai-native"
    OPENAI = "openai"
    # Future constants for native providers:
    # CLAUDE_NATIVE = "claude-native"
    GOOGLE = "google"
    GOOGLE_NATIVE = "google-native"
    SARVAMAI_NATIVE = "sarvamai-native"

    _registry: dict[str, type[BaseProvider]] | None = None

    @classmethod
    def _get_registry(cls) -> dict[str, type[BaseProvider]]:
        if cls._registry is None:
            cls._registry = _build_registry()
        return cls._registry

    @classmethod
    def get_provider_class(cls, provider_type: str) -> type[BaseProvider]:
        """Return the provider class for a given name."""
        registry = cls._get_registry()
        provider = registry.get(provider_type)
        if not provider:
            raise ValueError(
                f"Provider '{provider_type}' is not supported. "
                f"Supported providers: {', '.join(registry.keys())}"
            )
        return provider

    @classmethod
    def supported_providers(cls) -> list[str]:
        """Return a list of supported provider names."""
        return list(cls._get_registry().keys())


def get_llm_provider(
    session: Session, provider_type: str, project_id: int, organization_id: int
) -> BaseProvider:
    from app.crud.credentials import get_provider_credential

    provider_class = LLMProvider.get_provider_class(provider_type)

    # e.g., "openai-native" → "openai", "claude-native" → "claude"
    credential_provider = provider_type.replace("-native", "")

    credentials = get_provider_credential(
        session=session,
        provider=credential_provider,
        project_id=project_id,
        org_id=organization_id,
    )

    if not credentials:
        raise ValueError(
            f"Credentials for provider '{credential_provider}' not configured for this project."
        )

    try:
        client = provider_class.create_client(credentials=credentials)
        return provider_class(client=client)
    except ValueError:
        # Re-raise ValueError for credential/configuration errors
        raise
    except Exception as e:
        logger.error(f"Failed to initialize {provider_type} client: {e}", exc_info=True)
        raise RuntimeError(f"Could not connect to {provider_type} services.")
