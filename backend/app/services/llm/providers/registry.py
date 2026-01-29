import logging

from sqlmodel import Session
from openai import OpenAI

from app.crud import get_provider_credential
from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.openai import OpenAIProvider


logger = logging.getLogger(__name__)


class LLMProvider:
    OPENAI_NATIVE = "openai-native"
    OPENAI = "openai"
    # Future constants for native providers:
    # CLAUDE_NATIVE = "claude-native"
    # GEMINI_NATIVE = "gemini-native"

    _registry: dict[str, type[BaseProvider]] = {
        OPENAI_NATIVE: OpenAIProvider,
        OPENAI: OpenAIProvider,
        # Future native providers:
        # CLAUDE_NATIVE: ClaudeProvider,
        # GEMINI_NATIVE: GeminiProvider,
    }

    @classmethod
    def get(cls, name: str) -> type[BaseProvider]:
        """Return the provider class for a given name."""
        provider = cls._registry.get(name)
        if not provider:
            raise ValueError(
                f"Provider '{name}' is not supported. "
                f"Supported providers: {', '.join(cls._registry.keys())}"
            )
        return provider

    @classmethod
    def supported_providers(cls) -> list[str]:
        """Return a list of supported provider names."""
        return list(cls._registry.keys())


def get_llm_provider(
    session: Session, provider_type: str, project_id: int, organization_id: int
) -> BaseProvider:
    provider_class = LLMProvider.get(provider_type)

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

    if provider_type == LLMProvider.OPENAI_NATIVE:
        if "api_key" not in credentials:
            raise ValueError("OpenAI credentials not configured for this project.")
        client = OpenAI(api_key=credentials["api_key"])
    else:
        logger.error(
            f"[get_llm_provider] Unsupported provider type requested: {provider_type}"
        )
        raise ValueError(f"Provider '{provider_type}' is not supported.")

    return provider_class(client=client)
