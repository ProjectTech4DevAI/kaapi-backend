import logging
from sqlmodel import Session

from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.open_ai import OpenAIProvider
from app.services.llm.providers.google_aistudio import GoogleAIProvider
from app.services.llm.providers.sarvam_ai import SarvamAIProvider
from app.services.llm.providers.eleven_ai import ElevenlabsAIProvider
from app.services.llm.providers.claude import ClaudeProvider
from app.services.llm.providers.google_ai import GoogleVertexAIProvider

logger = logging.getLogger(__name__)


class LLMProvider:
    OPENAI = "openai"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    GOOGLE_AISTUDIO = "google-aistudio"
    GOOGLE_AISTUDIO_NATIVE = "google-aistudio-native"
    OPENAI_NATIVE = "openai-native"
    GOOGLE_NATIVE = "google-native"
    SARVAMAI_NATIVE = "sarvamai-native"
    ELEVENLABS_NATIVE = "elevenlabs-native"
    ANTHROPIC_NATIVE = "anthropic-native"

    _registry: dict[str, type[BaseProvider]] = {
        OPENAI: OpenAIProvider,
        GOOGLE: GoogleVertexAIProvider,
        SARVAMAI: SarvamAIProvider,
        ELEVENLABS: ElevenlabsAIProvider,
        ANTHROPIC: ClaudeProvider,
        GOOGLE_AISTUDIO: GoogleAIProvider,
        GOOGLE_AISTUDIO_NATIVE: GoogleAIProvider,
        OPENAI_NATIVE: OpenAIProvider,
        GOOGLE_NATIVE: GoogleVertexAIProvider,
        SARVAMAI_NATIVE: SarvamAIProvider,
        ELEVENLABS_NATIVE: ElevenlabsAIProvider,
        ANTHROPIC_NATIVE: ClaudeProvider,
    }

    @classmethod
    def get_provider_class(cls, provider_type: str) -> type[BaseProvider]:
        """Return the provider class for a given name."""
        provider = cls._registry.get(provider_type)
        if not provider:
            raise ValueError(
                f"Provider '{provider_type}' is not supported. "
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

    # Pass through whatever the DB returned (including None/empty). Providers
    # that support platform-default fallbacks (e.g. google) handle the
    # empty case themselves in create_client; others raise.
    if not credentials and credential_provider != "google":
        raise ValueError(
            f"Credentials for provider '{credential_provider}' not configured for this project."
        )
    credentials = credentials or {}

    try:
        client = provider_class.create_client(credentials=credentials)
        return provider_class(client=client)
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to initialize {provider_type} client: {e}", exc_info=True)
        raise RuntimeError(f"Could not connect to {provider_type} services.")
