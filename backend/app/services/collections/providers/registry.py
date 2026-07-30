import logging

from google import genai
from sqlmodel import Session
from openai import OpenAI

from app.crud import get_provider_credential
from app.crud.rag.open_ai import OPENAI_TIMEOUT_SECONDS
from app.services.collections.providers.base import BaseProvider
from app.services.collections.providers.gemini import GeminiAIStudioProvider
from app.services.collections.providers.openai import OpenAIProvider


logger = logging.getLogger(__name__)


class LLMProvider:
    OPENAI = "openai"
    GOOGLE_AISTUDIO = "google-aistudio"
    # Future constants for providers:
    # ANTHROPIC = "ANTHROPIC"

    _registry: dict[str, type[BaseProvider]] = {
        OPENAI: OpenAIProvider,
        GOOGLE_AISTUDIO: GeminiAIStudioProvider,
        # Future providers:
        # ANTHROPIC: BedrockProvider,
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
    session: Session, provider: str, project_id: int, organization_id: int
) -> BaseProvider:
    provider_class = LLMProvider.get(provider)

    credentials = get_provider_credential(
        session=session,
        provider=provider,
        project_id=project_id,
        org_id=organization_id,
    )

    if not credentials:
        raise ValueError(
            f"Credentials for provider '{provider}' not configured for this project."
        )

    if provider == LLMProvider.OPENAI:
        if "api_key" not in credentials:
            raise ValueError("OpenAI credentials not configured for this project.")
        client = OpenAI(
            api_key=credentials["api_key"],
            # SDK retries off; tenacity in OpenAIVectorStoreCrud is the sole retry layer.
            max_retries=0,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
    elif provider == LLMProvider.GOOGLE_AISTUDIO:
        if "api_key" not in credentials:
            raise ValueError(
                "Google AI Studio credentials not configured for this project."
            )
        client = genai.Client(api_key=credentials["api_key"])
    else:
        logger.warning(
            f"[get_llm_provider] Unsupported provider type requested: {provider}"
        )
        raise ValueError(f"Provider '{provider}' is not supported.")

    return provider_class(client=client)
