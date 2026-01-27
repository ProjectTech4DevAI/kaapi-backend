import os
from dotenv import load_dotenv
import logging

from sqlmodel import Session
from openai import OpenAI

from app.crud import get_provider_credential
from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.oai import OpenAIProvider
from app.services.llm.providers.gai import GoogleAIProvider

from google.genai.types import GenerateContentConfig

# temporary import

from app.models.llm import (
    NativeCompletionConfig,
    LLMCallResponse,
    QueryParams,
    LLMOutput,
    LLMResponse,
    Usage,
)

load_dotenv()

logger = logging.getLogger(__name__)


class LLMProvider:
    OPENAI_NATIVE = "openai-native"
    # Future constants for native providers:
    # CLAUDE_NATIVE = "claude-native"
    GOOGLE_NATIVE = "google-native"

    _registry: dict[str, type[BaseProvider]] = {
        OPENAI_NATIVE: OpenAIProvider,
        # Future native providers:
        # CLAUDE_NATIVE: ClaudeProvider,
        GOOGLE_NATIVE: GoogleAIProvider,
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
    provider_class = LLMProvider.get_provider_class(provider_type)

    # e.g "openai-native" -> "openai", "claude-native" -> "claude"
    credential_provider = provider_type.replace("-native", "")

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


# ad hoc testing code
if __name__ == "__main__":
    # 1. Simulate environment/credentials
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        print("Set GEMINI_API_KEY environment variable first.")
        exit(1)

    # This dictionary mimics what get_provider_credential would return from the DB
    mock_credentials = {"api_key": GEMINI_KEY}

    # 2. Idiomatic Initialization via Registry
    provider_type = "google-native"
    # provider_type=LLMProvider.get_provider_class(provider_type="GOOGLE-NATIVE")

    print(f"Initializing provider: {provider_type}...")

    # This block mimics the core logic of your get_llm_provider function
    ProviderClass = LLMProvider.get_provider_class(provider_type)
    client = ProviderClass.create_client(credentials=mock_credentials)
    instance = ProviderClass(client=client)

    # 3. Setup Config and Query
    test_config = NativeCompletionConfig(
        provider="google-native",
        type="stt",
        params={
            "model": "gemini-2.5-pro",
            "instructions": "Please transcribe this audio accurately.",
        },
    )

    test_query = QueryParams(
        input="/Users/prajna/Desktop/personal/projects/software/Syspin_Hackathon_api_server/wav_files/1253534463206645.wav"  # Ensure this file exists in your directory
    )

    # 4. Execution
    print("Executing STT...")
    result, error = instance.execute(completion_config=test_config, query=test_query)

    if error:
        print(f"Error: {error}")
    else:
        print(f"Result: {result}")
