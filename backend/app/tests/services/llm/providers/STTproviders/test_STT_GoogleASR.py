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
from app.services.llm.providers.gasr  import GoogleSpeechtoTextAIProvider

from app.services.llm.providers.tests_data import  mydata

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


def create_audio_file_from_mydata(audio_data: bytes) -> str:
    """
    Creates a temporary WAV file from provided audio data (bytes)
    and returns the path to the file.
    """
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_audio_file.write(audio_data)
    temp_audio_file.close()
    print(f"Created temporary audio file at: {temp_audio_file.name}")
    return temp_audio_file.name


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

    file_path = 'probable-splice-440907-h2-56b4982d92cd.json'
    file_path = 'app/services/llm/providers/probable-splice-440907-h2-56b4982d92cd.json'
    file_path = 'app/tests/services/llm/providers/STTproviders/probable-splice-440907-h2-56b4982d92cd.json'

    print(f"Current working directory: {os.getcwd()}")
    if os.path.exists(file_path):
        print(f"Nan..The file '{file_path}' exists.")
    else:
        print(f"Nan..The file '{file_path}' does not exist.")

    # Update the LLMProvider registry to include GoogleSpeechtoTextAIProvider
    if "google-speech-to-text-native" not in LLMProvider._registry:
        LLMProvider._registry["google-speech-to-text-native"] = GoogleSpeechtoTextAIProvider
        print("GoogleSpeechtoTextAIProvider registered successfully in LLMProvider.")
    else:
        print("GoogleSpeechtoTextAIProvider was already registered.")

    print(f"Supported providers after update: {LLMProvider.supported_providers()}")

   # key_filename = "probable-splice-440907-h2-56b4982d92cd.json"
    key_filename = file_path

    audio_file_path = create_audio_file_from_mydata(mydata)

    project_id = 'probable-splice-440907-h2'


    # 1. Prepare credentials for client creation
    credentials = {
        "key_filename": key_filename,
        "project_id": project_id,
        "api_endpoint": "asia-northeast1" # Specify the endpoint based on model availability
    }

    # 2. Create the Speech Client using the static method
    try:
        # create_client now only returns the client object
        speech_client_obj = GoogleSpeechtoTextAIProvider.create_client(credentials)
    except ValueError as e:
        print(f"Error creating client: {e}")
        speech_client_obj = None

    # Check if the project_id is available in credentials, which is now expected
    if speech_client_obj and credentials.get('project_id'):
        # 3. Instantiate the provider, passing the client object and the project_id explicitly
        provider = GoogleSpeechtoTextAIProvider(client=speech_client_obj, project_id=credentials['project_id'])

        # 4. Define completion configuration (similar to NativeCompletionConfig)
        completion_config = NativeCompletionConfig(
            provider="google-speech-to-text-native", # Use the registered provider name
            type="stt",
            params={
                "model": "chirp_3",
                "language_codes": ["auto"], # Using 'ta-IN' based on the audio content, 'auto' is also an option
                "recognizer_id": "_", # Use underscore for default recognizer if not specified
                "location": "asia-northeast1", # Must match the api_endpoint for the client
                "project_id": credentials['project_id'] # Pass the project ID for consistency if needed in params
            }
        )

        test_query = QueryParams(
            input={"type": "text", "content": "Transcription request"}
        )

        # 5. Execute STT
        # Use the 'audio_file_path' variable created from 'mydata'
        print(audio_file_path)
        response, error = provider.execute(
            completion_config=completion_config,
            query=test_query, # dummy QueryParams for STT
            resolved_input=audio_file_path, # Use the path to the temporary audio file
            include_provider_raw_response=False
        )

        if error:
            print(f"STT Error: {error}")
        elif response:
            print(f"Transcript from GoogleSpeechtoTextAIProvider: {response.response.output.text}")
            print(f"Usage: Input Tokens={response.usage.input_tokens}, Output Tokens={response.usage.output_tokens}, Total Tokens={response.usage.total_tokens}")
    else:
        print("Could not initialize GoogleSpeechtoTextAIProvider: Client or project ID missing.")
