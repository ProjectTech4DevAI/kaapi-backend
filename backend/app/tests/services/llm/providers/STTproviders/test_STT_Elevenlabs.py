import os
from dotenv import load_dotenv
import logging
import io

from sqlmodel import Session
from openai import OpenAI

from app.crud import get_provider_credential
from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.oai import OpenAIProvider
from app.services.llm.providers.gai import GoogleAIProvider
from app.services.llm.providers.eai import ElevenLabsAIProvider
from app.tests.services.llm.providers.STTproviders.test_data_speechsamples import mydata



from google.genai.types import GenerateContentConfig
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



# ad hoc testing code for ElevenLabsAIProvider
if __name__ == "__main__":
    # 1. Simulate environment/credentials
    # ELEVENLABS_API_KEY must be set in Colab secrets
    # This variable should be correctly populated after setting the secret
    ElevenLabs_API_KEY = 'sk_7428916b02e15358d961001cc1c17a27a5f7e4a6ce942d48'
    try:
        ELEVENLABS_API_KEY = ElevenLabs_API_KEY
    except Exception as e:
        print(f"Error retrieving ELEVENLABS_API_KEY from secrets: {e}")
        print("Please ensure ELEVENLABS_API_KEY is set in .")
        exit(1)

    if not ELEVENLABS_API_KEY:
        print("ELEVENLABS_API_KEY is not set. Please set it in Colab secrets and re-run.")
        exit(1)

    # This dictionary mimics what get_provider_credential would return from the DB
    mock_credentials_elevenlabs = {"api_key": ELEVENLABS_API_KEY}

    # 2. Idiomatic Initialization via Registry (using the ElevenLabsAIProvider class directly)
    provider_type_elevenlabs = "elevenlabs-native" # Conceptual identifier

    print(f"Initializing provider: {provider_type_elevenlabs}...")

    try:
        # Directly create client and instance for demonstration
        elevenlabs_client = ElevenLabsAIProvider.create_client(credentials=mock_credentials_elevenlabs)
        elevenlabs_instance = ElevenLabsAIProvider(client=elevenlabs_client)
        print("ElevenLabsAIProvider instance created successfully.")
    except Exception as e:
        print(f"Error initializing ElevenLabsAIProvider: {e}")
        exit(1)

    # 3. Prepare audio data
    # 'mydata' is assumed to be bytes from the previous context.
    # Wrap it in BytesIO as ElevenLabsAIProvider expects file-like object or path.
    audio_input_object_elevenlabs = None
    try:
        if 'mydata' not in locals() or not isinstance(mydata, bytes):
            print("mydata (audio bytes) not found. Creating dummy audio data for demonstration...")
            # In a real use case, you'd load your actual audio data here.
            # For example, by fetching from a URL and wrapping in BytesIO
            import requests
            audio_url = "https://storage.googleapis.com/eleven-public-cdn/audio/marketing/nicole.mp3" # A sample audio file
            response = requests.get(audio_url)
            mydata = response.content # Store dummy data in mydata
            print("Dummy audio data created from a sample URL.")

        audio_input_object_elevenlabs = io.BytesIO(mydata)
        audio_input_object_elevenlabs.seek(0) # Ensure cursor is at the beginning
        print("Audio data prepared for ElevenLabsAIProvider.")

        # 4. Setup Config and Query for ElevenLabsAIProvider
        elevenlabs_config = NativeCompletionConfig(
            provider="google-speech-to-text-native", # Use accepted literal for STT type in the system
            type="stt",                             # Use 'stt' as the accepted type literal
            params={                                # ElevenLabs-specific parameters nested under 'params'
                "completion_type": "stt",           # Indicate STT specifically for the provider's internal logic
                "model_name": "scribe_v2",        # ElevenLabs specific model
                "input_language": None,           # ElevenLabs specific language (None for auto-detect)
                "diarize": True,                    # ElevenLabs specific feature
                "tag_audio_events": True          # ElevenLabs specific feature
            }
        )

        elevenlabs_query = QueryParams(
            input={"type": "text", "content": "Transcription request for ElevenLabs"} # Required for QueryParams
        )

        # 5. Execution
        print("Executing STT with ElevenLabsAIProvider...")

        result_elevenlabs, error_elevenlabs = elevenlabs_instance.execute(
            completion_config=elevenlabs_config,
            query=elevenlabs_query,
            resolved_input=audio_input_object_elevenlabs, # Pass the BytesIO object
            include_provider_raw_response=True # Set to True to see ElevenLabs' raw response
        )

        # 6. Display Results
        if error_elevenlabs:
            print(f"\nElevenLabs Transcription Error: {error_elevenlabs}")
        else:
            print("\nElevenLabs Transcription Result:")
            print(f"Model: {result_elevenlabs.model}")
            print(f"Text: {result_elevenlabs.text}")
            print(f"Usage (Word Count): {result_elevenlabs.usage.get('word_count', 'N/A')}")
            if result_elevenlabs.provider_raw_response:
                print("\nElevenLabs Raw Response Snippet (first 2 words from transcription.words):")
                raw_words = result_elevenlabs.provider_raw_response.words
                if raw_words:
                    for word_info in raw_words[:2]:
                        print(f"  - '{word_info.text}' (Speaker: {word_info.speaker_id}, Start: {word_info.start:.2f}s, End: {word_info.end:.2f}s)")
                else:
                    print("  No word-level details in raw response.")

    finally:
        # No specific cleanup for BytesIO needed beyond scope exit, but good practice to show `finally`.
        if audio_input_object_elevenlabs and not audio_input_object_elevenlabs.closed:
            audio_input_object_elevenlabs.close()
            print("Cleaned up audio input object.")
