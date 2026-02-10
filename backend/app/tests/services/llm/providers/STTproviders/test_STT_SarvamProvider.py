import os
from dotenv import load_dotenv
import logging

from sqlmodel import Session
from openai import OpenAI

from app.crud import get_provider_credential
from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.oai import OpenAIProvider
from app.services.llm.providers.gai2 import GoogleAIProvider
from app.services.llm.providers.sai import SarvamAIProvider

from app.tests.services.llm.providers.STTproviders.test_data_speechsamples import mydata

import tempfile


# ad hoc testing code for SarvamAIProvider
import os
import tempfile

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
    OPENAI = "openai"
    # Future constants for native providers:
    # CLAUDE_NATIVE = "claude-native"
    GOOGLE_NATIVE = "google-native"

    _registry: dict[str, type[BaseProvider]] = {
        OPENAI_NATIVE: OpenAIProvider,
        OPENAI: OpenAIProvider,
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



if __name__ == "__main__":
    # 1. Simulate environment/credentials
    # SARVAM_API_KEY is already defined in the notebook
    SARVAM_API_KEY = "sk_lmsvfc31_On1bxqwDAqYZoijqBfblr3yf"  # for testing only

    if not SARVAM_API_KEY:
        print("SARVAM_API_KEY is not set.")
        exit(1)

    # This dictionary mimics what get_provider_credential would return from the DB
    mock_credentials = {"api_key": SARVAM_API_KEY}

    # 2. Idiomatic Initialization via Registry


    
    provider_type = "sarvamai-native"
    # Adding SarvamAIProvider to the registry
    if "sarvamai-native" not in LLMProvider._registry:
        LLMProvider._registry["sarvamai-native"] = SarvamAIProvider
        print("SarvamAIProvider registered successfully in LLMProvider.")
    else:
        print("SarvamAIProvider was already registered.")


    print(f"Initializing provider: {provider_type}...")

    # This block mimics the core logic of your get_llm_provider function
    ProviderClass = LLMProvider.get_provider_class(provider_type)
    client = ProviderClass.create_client(credentials=mock_credentials)
    instance = ProviderClass(client=client)

    # Save the base64 decoded audio data to a temporary file
    temp_audio_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_file.write(mydata)
            temp_audio_file_path = temp_audio_file.name

        # 3. Setup Config and Query
        test_config = NativeCompletionConfig(
            provider="sarvamai-native",
            type="stt",
            params={
                #"model": "saarika:v2.5", # Using SarvamAI's model for STT
                "model": "saaras:v3", # Using SarvamAI's model for STT
                "input_language":"unknown",  # Let SarvamAI auto-detect the language with 'unknown' or specify if known (e.g., "ta-IN", "hi-IN")
                # SarvamAI's transcribe method doesn't directly take 'prompt instructions' like LLMs 
            },
        )

 
        test_query = QueryParams(
            input={"type": "text", "content": "Transcription request"}
        )

        # 4. Execution
        print("Executing STT with SarvamAIProvider...")
        # For STT, resolved_input needs to be the file path
        result, error = instance.execute(completion_config=test_config, query=test_query, resolved_input=temp_audio_file_path)

        if error:
            print(f"Error: {error}")
        else:
            print(f"\n--- SarvamAI STT Result ---")
            print(f"Transcribed Text: {result.response.output.text}")
            print(f"Provider Model: {result.response.model}")
            print("\n--- Usage Information ---")
            print(f"Input Tokens: {result.usage.input_tokens}")
            print(f"Output Tokens: {result.usage.output_tokens}")
            print(f"Total Tokens: {result.usage.total_tokens}")
            if result.usage.reasoning_tokens:
                print(f"Reasoning Tokens: {result.usage.reasoning_tokens}")
            # Uncomment to see the raw response:
            # import json
            # print("\n--- Raw Provider Response ---")
            # print(result.provider_raw_response)

    finally:
        # Clean up the temporary file
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            os.remove(temp_audio_file_path)
            print(f"Cleaned up temporary file: {temp_audio_file_path}")
 