import logging
import os
import asyncio
from dotenv import load_dotenv

from sqlmodel import Session
from typing import Any, Dict, Type
from app.crud import get_provider_credential
from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.oai import OpenAIProvider
from app.services.llm.providers.gai import GoogleAIProvider
from app.models.llm import KaapiCompletionConfig, QueryParams, KaapiLLMParams

load_dotenv()
logger = logging.getLogger(__name__)


class LLMProvider:
    OPENAI_NATIVE = "openai-native"
    OPENAI = "openai"
    GOOGLE = "google"
    # Future constants for native providers:
    # CLAUDE_NATIVE = "claude-native"
    # GEMINI_NATIVE = "gemini-native"

    _registry: dict[str, type[BaseProvider]] = {
        OPENAI_NATIVE: OpenAIProvider,
        OPENAI: OpenAIProvider,
        GOOGLE: GoogleAIProvider
        # Future native providers:
        # CLAUDE_NATIVE: ClaudeProvider,
        # GEMINI_NATIVE: GeminiProvider,
    }

    @classmethod
    def get_provider_class(cls, provider_type: str) -> Type[BaseProvider]:
        """Return the provider class for a given provider_type."""
        # provider = cls._registry.get(provider_type)
        if provider_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Provider '{provider_type}' is not supported. "
                f"Supported providers: {available}"
            )
        return cls._registry[provider_type]

    @classmethod
    def supported_providers(cls) -> list[str]:
        """Return a list of supported provider names."""
        return list(cls._registry.keys())


# find out the llm_provider for an org filtered by project and provider type
def get_llm_provider(
    session: Session, provider_type: str, project_id: int, organization_id: int
) -> BaseProvider:
    """
    Orchestrator function that:
    1. Finds the right Provider Class.
    2. Fetches the correct credentials for the Project/Org.
    3. Initializes the client via the Provider's internal factory.
    4. Returns a ready-to-use Provider instance.
    """
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
        # build the client
        client = provider_class.create_client(credentials=credentials)  # type: ignore

        return provider_class(client=client)
    except Exception as e:
        logger.error(f"Failed to initialize {provider_type} client: {e}", exc_info=True)
        raise RuntimeError(f"Could not connect to {provider_type} services.")


def test_initialization():
    google_creds = {"api_key": os.getenv("GEMINI_API_KEY")}
    oai_creds = {"api_key": os.getenv("OPENAI_API_KEY")}

    oa_provider = OpenAIProvider(client=OpenAIProvider.create_client(oai_creds))

    google_provider = GoogleAIProvider(
        client=GoogleAIProvider.create_client(google_creds)
    )

    test_query = QueryParams(
        input="Explain the concept of 'Recursion' in one  sentence"
    )

    # OpenAI Config
    oa_config = KaapiCompletionConfig(
        provider="openai",
        type="text",
        params=KaapiLLMParams(
            model="gpt-4o-mini",
            temperature=0.5,
        ),
    )

    # Google Config (Optimized for Gemini 3 Flash)
    google_config = KaapiCompletionConfig(
        provider="google",
        type="text",
        params=KaapiLLMParams(
            model="gemini-3-flash-preview", temperature=0.5, reasoning="low"
        ),
    )

    # 4. Execute and Print Results
    print(f"{'='*20} Testing OpenAI {'='*20}")
    oa_resp, oa_err = oa_provider.execute(oa_config, test_query)
    if oa_resp:
        print(f"Response: {oa_resp.response.output.text}")
        print(f"Usage: {oa_resp.usage}")
    else:
        print(f"Error: {oa_err}")

    print(f"\n{'='*20} Testing Google Gemini {'='*20}")
    gem_resp, gem_err = google_provider.execute(google_config, test_query)
    if gem_resp:
        print(f"Response: {gem_resp.response}")
        print(f"Usage: {gem_resp.usage}")
    else:
        print(f"Error: {gem_err}")


if __name__ == "__main__":
    test_initialization()
