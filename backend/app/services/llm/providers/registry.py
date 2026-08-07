import logging
from sqlmodel import Session

from app.core.config import settings
from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.open_ai import OpenAIProvider
from app.services.llm.providers.google_aistudio import GoogleAIProvider
from app.services.llm.providers.google_ai import GoogleVertexAIProvider
from app.services.llm.providers.sarvam_ai import SarvamAIProvider
from app.services.llm.providers.eleven_ai import ElevenlabsAIProvider
from app.services.llm.providers.claude import ClaudeProvider

logger = logging.getLogger(__name__)


class LLMProvider:
    OPENAI = "openai"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"
    ANTHROPIC = "anthropic"

    # Platform-routed Google. Resolved via GEMINI_DEFAULT_INFERENCE_ROUTE.
    GOOGLE = "google"
    GOOGLE_NATIVE = "google-native"
    # Explicit Google backends. Bypass env routing; use caller's own creds.
    GOOGLE_VERTEX = "google-vertex"
    GOOGLE_VERTEX_NATIVE = "google-vertex-native"
    GOOGLE_AISTUDIO = "google-aistudio"
    GOOGLE_AISTUDIO_NATIVE = "google-aistudio-native"

    OPENAI_NATIVE = "openai-native"
    SARVAMAI_NATIVE = "sarvamai-native"
    ELEVENLABS_NATIVE = "elevenlabs-native"
    ANTHROPIC_NATIVE = "anthropic-native"

    _registry: dict[str, type[BaseProvider]] = {
        OPENAI: OpenAIProvider,
        SARVAMAI: SarvamAIProvider,
        ELEVENLABS: ElevenlabsAIProvider,
        ANTHROPIC: ClaudeProvider,
        GOOGLE: GoogleVertexAIProvider,  # placeholder; env decides at resolve time
        GOOGLE_NATIVE: GoogleVertexAIProvider,
        GOOGLE_VERTEX: GoogleVertexAIProvider,
        GOOGLE_VERTEX_NATIVE: GoogleVertexAIProvider,
        GOOGLE_AISTUDIO: GoogleAIProvider,
        GOOGLE_AISTUDIO_NATIVE: GoogleAIProvider,
        OPENAI_NATIVE: OpenAIProvider,
        SARVAMAI_NATIVE: SarvamAIProvider,
        ELEVENLABS_NATIVE: ElevenlabsAIProvider,
        ANTHROPIC_NATIVE: ClaudeProvider,
    }

    _GOOGLE_ROUTED = {GOOGLE, GOOGLE_NATIVE}

    @classmethod
    def get_provider_class(cls, provider_type: str) -> type[BaseProvider]:
        if provider_type in cls._GOOGLE_ROUTED:
            # Only an explicit "vertex" opts into the new route; anything else
            # (default, unset, typo) keeps today's aistudio behavior.
            route = settings.GEMINI_DEFAULT_INFERENCE_ROUTE
            return GoogleVertexAIProvider if route == "vertex" else GoogleAIProvider
        provider = cls._registry.get(provider_type)
        if not provider:
            raise ValueError(
                f"Provider '{provider_type}' is not supported. "
                f"Supported providers: {', '.join(cls._registry.keys())}"
            )
        return provider

    @classmethod
    def supported_providers(cls) -> list[str]:
        return list(cls._registry.keys())


def get_llm_provider(
    session: Session, provider_type: str, project_id: int, organization_id: int
) -> BaseProvider:
    from app.crud.credentials import get_provider_credential

    provider_class = LLMProvider.get_provider_class(provider_type)
    is_vertex_routed = (
        provider_type in LLMProvider._GOOGLE_ROUTED
        and provider_class is GoogleVertexAIProvider
    )

    if is_vertex_routed:
        # Platform-routed vertex: tenant row, else platform GCP settings.
        credential_chain = [LLMProvider.GOOGLE_VERTEX]
    elif provider_type in LLMProvider._GOOGLE_ROUTED:
        # Platform-routed aistudio keeps today's `google` row semantics verbatim.
        credential_chain = [LLMProvider.GOOGLE]
    elif provider_type in (
        LLMProvider.GOOGLE_AISTUDIO,
        LLMProvider.GOOGLE_AISTUDIO_NATIVE,
    ):
        # Explicit aistudio falls back to the legacy `google` row.
        credential_chain = [LLMProvider.GOOGLE_AISTUDIO, LLMProvider.GOOGLE]
    else:
        credential_chain = [provider_type.replace("-native", "")]

    credentials = None
    for credential_provider in credential_chain:
        credentials = get_provider_credential(
            session=session,
            provider=credential_provider,
            project_id=project_id,
            org_id=organization_id,
        )
        if credentials:
            break

    if not credentials and not is_vertex_routed:
        raise ValueError(
            f"Credentials for provider '{credential_chain[0]}' not configured for this project."
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
