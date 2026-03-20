import logging
import time
from sqlmodel import Session

from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.oai import OpenAIProvider
from app.services.llm.providers.gai import GoogleAIProvider
from app.services.llm.providers.sai import SarvamAIProvider
from app.services.llm.providers.eai import ElevenlabsAIProvider

logger = logging.getLogger(__name__)


class LLMProvider:
    OPENAI = "openai"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"
    GOOGLE = "google"
    # Future constants for native providers:
    # CLAUDE_NATIVE = "claude-native"
    OPENAI_NATIVE = "openai-native"
    GOOGLE_NATIVE = "google-native"
    SARVAMAI_NATIVE = "sarvamai-native"
    ELEVENLABS_NATIVE = "elevenlabs-native"

    _registry: dict[str, type[BaseProvider]] = {
        OPENAI: OpenAIProvider,
        GOOGLE: GoogleAIProvider,
        SARVAMAI: SarvamAIProvider,
        ELEVENLABS: ElevenlabsAIProvider,
        # Future native providers:
        # CLAUDE_NATIVE: ClaudeProvider,
        OPENAI_NATIVE: OpenAIProvider,
        GOOGLE_NATIVE: GoogleAIProvider,
        SARVAMAI_NATIVE: SarvamAIProvider,
        ELEVENLABS_NATIVE: ElevenlabsAIProvider,
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

    provider_start = time.perf_counter()

    t_class_start = time.perf_counter()
    provider_class = LLMProvider.get_provider_class(provider_type)
    t_class = (time.perf_counter() - t_class_start) * 1000

    # e.g., "openai-native" → "openai", "claude-native" → "claude"
    credential_provider = provider_type.replace("-native", "")

    t_cred_start = time.perf_counter()
    credentials = get_provider_credential(
        session=session,
        provider=credential_provider,
        project_id=project_id,
        org_id=organization_id,
    )
    t_cred = (time.perf_counter() - t_cred_start) * 1000

    if not credentials:
        raise ValueError(
            f"Credentials for provider '{credential_provider}' not configured for this project."
        )

    try:
        t_client_start = time.perf_counter()
        client = provider_class.create_client(credentials=credentials)
        provider_instance = provider_class(client=client)
        t_client = (time.perf_counter() - t_client_start) * 1000

        total_time = (time.perf_counter() - provider_start) * 1000

        logger.info(
            f"[TIMING] get_llm_provider | provider={provider_type} | "
            f"get_class={t_class:.2f}ms, get_creds={t_cred:.2f}ms, "
            f"create_client={t_client:.2f}ms, total={total_time:.2f}ms"
        )

        return provider_instance
    except ValueError:
        # Re-raise ValueError for credential/configuration errors
        raise
    except Exception as e:
        logger.error(f"Failed to initialize {provider_type} client: {e}", exc_info=True)
        raise RuntimeError(f"Could not connect to {provider_type} services.")
