Integration of Gemini Pro 2.5 Speech To Text Model with LLM Call Endpoint.

Inside app/services/llm/providers/gai.py the "stt" gemini code is implemented

```
import logging

import openai
from openai import OpenAI
from google import genai
from google.genai.types import GenerateContentResponse
from typing import Any, Tuple
from openai.types.responses.response import Response

from app.models.llm import (
    NativeCompletionConfig,
    LLMCallResponse,
    QueryParams,
    LLMOutput,
    LLMResponse,
    Usage,
)
from app.services.llm.providers.base import BaseProvider


logger = logging.getLogger(__name__)

#TODO fix circular import issue with GoogleAIProvider and OpenAIProvider
class GoogleAIProvider(BaseProvider):
    def __init__(self, client: genai.Client):
        """Initialize OpenAI provider with client.

        Args:
            client: OpenAI client instance
        """
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        if "api_key" not in credentials:
            return f"Gemini API Key not configured."
        return genai.Client(api_key=credentials["api_key"])

    def _parse_input(self, query_input, completion_type, provider) -> str:
        if completion_type == "stt":
            if isinstance(query_input, str):
                return query_input
            else:
                raise ValueError(f"{provider} STT require file path")

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        include_provider_raw_response: bool = False,
    ) -> tuple[Any, str | None]:
        response: Response | None = None
        error_message: str | None = None

        try:
            completion_type = completion_config.type
            provider = completion_config.provider

            generation_params = completion_config.params
            if completion_type == "stt":
                # query.input would be an audio_file object, or pathname

                parsed_input = self._parse_input(
                    query_input=query.input,
                    completion_type=completion_type,
                    provider=provider,
                )

                model = generation_params.get("model")
                instructions = generation_params.get("instructions")

                if not model:
                    return None, "Missing 'model' in native params"
                parsed_input = self._parse_input(query.input, completion_type, provider)

                gemini_file = self.client.files.upload(file=parsed_input)

                contents = []

                if instructions:
                    contents.append(instructions)
                contents.append(gemini_file)
                response :GenerateContentResponse= self.client.models.generate_content(
                    model=model, contents=contents
                )

                llm_response=LLMCallResponse(
                    response=LLMResponse(
                        provider_response_id=response.response_id,
                        model=response.model_version,
                        provider=provider,
                        output=LLMOutput(
                            text=response.text
                        )
                    ),
                    usage=Usage(
                        input_tokens=response.usage_metadata.prompt_token_count,
                        output_tokens=response.usage_metadata.candidates_token_count,
                        total_tokens=response.usage_metadata.total_token_count,
                        reasoning_tokens=response.usage_metadata.thoughts_token_count
                    ),
                )

                if include_provider_raw_response:
                    llm_response.provider_raw_response=response.model_dump()

                logger.info(
                    f"[OpenAIProvider.execute] Successfully generated response: {response.response_id}"
                )

                return llm_response, None

        except TypeError as e:
            # handle unexpected arguments gracefully
            error_message = f"Invalid or unexpected parameter in Config: {str(e)}"
            return None, error_message

        except Exception as e:
            error_message = "Unexpected error occurred"
            logger.error(
                f"[GoogleAIProvider.execute] {error_message}: {str(e)}", exc_info=True
            )
            return None, error_message

```

Go through the code inside app/models/llm/request.py and complete the llm call table.

Spec

llm_call Table
id (required, PK): Unique identifier for the response
input (required, string | bytes | file_path): User input - can be text string, binary data, or a pointer to the file object in case of Gemini Files API.
input_type (required, string):text, audio, image etc
output_type (optional, string): text, audio, image etc.
status: pending | processing | completed | failed
(status is already included in the ‘job’ table)
provider_response_id (optional, string): The original response ID from the provider (e.g., OpenAI's response ID)
conversation_id (optional, string): Identifier linking this response to its conversation thread
auto_create (optional, boolean): Whether to automatically create a new conversation if conversation_id doesn't exist. OpenAI specific.
config (optional, JSONB):
config_id (optional, FK): Foreign key reference to the configuration ID to use for this request
config_version (optional, integer): Version number of the configuration to use
config_blob(optional, JSONB): alternative to config_id and version.Schema corresponds to native completion configuration.

provider (required, string): AI provider that generated the response (e.g., "openai", "google", "anthropic")
model (required, string): Specific model used to generate the response (e.g., "gpt-4o", "gemini-2.5-pro")
content (required, JSONB): Response content with structure:
{ text:”This is the lorem text},
{ audio_bytes: encoded.mp3”},
{image: generated.png},
usage (optional, JSONB):
 	Input_tokens (optional, int):
	output_tokens (optional, int):
reasoning_tokens (optional, int):


class LlmCall(SQLModel, table=True):
    __tablename__ = "llm_call"
    __table_args__ = (
        UniqueConstraint(
            "config_id", "version", name="uq_config_version_config_id_version"
        ),
        Index(
            "idx_config_version_config_id_version_active",
            "config_id",
            "version",
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the configuration version"},

    )

    job_id:str =Field(
        foreign_key="job.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the parent configuration"},
    )

    provider:Literal["openai", "google"]=Field(
        ...,
        description="LLM API Provider"

    )
    input:str =Field(
        ...,
        description="Plain text or a filepath for multimodal input."
    )

    input_type:Literal["text", "audio"]=Field(
        ...,
        description="Input type enum. Extensible for future use cases"
    )

    output_type:Literal["text", "audio"] | None=Field(
        default=None,
        description="LLM Call response output"
    )

    model:str=Field(
        ...,
        description="Specific model used e.g 'gpt-4o', 'gemini-2.5-pro'"
    )

    provider_response_id:str | None=Field(
        default=None,
        description="The original response ID from the provider (e.g., OpenAI's response ID)"
    )

    conversation_id:str| None=Field(
        default=None
        description="Identifier linking this response to its conversation thread"
    )
    config:JSONB | None=Field(
        default=None,
        description="Configuration used to make the LLM Call"
    )

    created_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the job was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the job was last updated"},
    )

    deleted_at: datetime | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Timestamp when the version was soft-deleted"},
    )

Additional configuration context
Backdrop:
For the configuration to take the multimodal nature of inputs into account, the fields key-value pairs inside the nested completion block will have to be purpose built for the type of llm-call (text-generation, audio-generation, image-generation etc) tasks at hand. For the configuration to make sure it does not break the llm/call API endpoint, relevant changes need to be made as well.

Also for the complete traceability of llm request-response(including multimodal), relevant objects need to be stored in their specific table as well.

Configuration Schemas:

The current provider-agnostic schema looks like this

Provider agnostic configuration schema (Current)
{
  "completion": {
    "provider": "openai",
    "type": "text",
    "params": {
      "model": "gpt-5",
      "instructions": "Answer the question clearly.",
	 "knowledge_base_ids": ["vs_12345"],
      "reasoning":"medium",
      "max_num_result" : 20,
      "temperature":0.7, (only one of reasoning or temperature based on model)
    }
  },
  "classifier": {},
  "input_guardrails": {},
  "output_guardrails": {}
}

The shortcomings of this config is

The configuration works for text-generation use cases only
// Text configuration

{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  “name”:”Openai Test Configuration”,
  “description”:”Lorem ipsum dolor sit amet”,
  "version": 1,
  "config_blob": {
    "completion": {
      "provider": "openai" | “google” | “anthropic”,
      "type": "text",
      "params": {
        "model": "gpt-4o",
        "instructions": "You are a helpful assistant",
        "knowledge_base_ids": ["vs_123", "vs_456"],
        "reasoning": "low" | "medium" | "high",
        "temperature": 0.7,
        "max_tokens": 1000,
	   “max_num_results:20,
        "provider_specific": {

		"openai": {},
          "google": {}

}
      }
    },

// for future extensions
  "classifier": {},
  "input_guardrails": {},
  "output_guardrails": {}

  }
}

// STT Configuration
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
 “name”:”Gemini Configuration”,
  “description”:”Lorem ipsum dolor sit amet”,

  "version": 2,
  "config_blob": {
    "completion": {
      "provider": "google",
      "type": "stt",
      "params": {
        "model": "gemini-2.5-pro",
        "instruction": "Transcribe the audio verbatim",
        "input_language": "hi",
        "output_language": "en", (#Useful for translation transcription in a single step)
        "response_format": "text"| “json”,
        "temperature": 0.7,
        "provider_specific": {
          "openai": {},
          "google": {}
        }
      }
    },
    "classifier": {},
    "input_guardrails": {},
    "output_guardrails": {}


  }
}

// TTS Configuration
{
  "id": "550e8400-e29b-41d4-a716-446655440002",
  "version": "1.0.0",
  "config_blob": {
    "completion": {
      "provider": "google",
      "type": "tts",
      "params": {
        "model": "gemini-2.5-pro-tts",
        "voice": "alloy",
        "language": "en-US",
        "response_format": "mp3" | “wav” | “ogg”,
        "speed": 1.0,
        "provider_specific": {
          "openai": {},
          "gemini": {
    "director_notes": "Speak with a professional, calm tone. Pause for 1 second between sentences.",

"response_modalities": ["AUDIO"] # example metadata for tracing

}
        }
      }
    },


    "classifier": {},
    "input_guardrails": {},
    "output_guardrails": {}

  }
}

Field Definitions
Top-Level Fields
id (required, string, UUID): Unique identifier for the configuration
name (required, string): Human-readable name for the configuration
description (optional, string): Description of the configuration's purpose
version (required, integer): Configuration version number
config_blob (required, object): Container for all configuration settings

config_blob Fields
completion (required, object): AI model behavior configuration
classifier (optional, object): For future use (planned for request classification and routing)
input_guardrails (optional, object): For future use (planned for input validation and safety checks)
output_guardrails (optional, object): For future use (planned for output filtering and safety checks)
completion Fields
provider (required, string): AI provider identifier
Allowed values: "openai", "google", "anthropic"
type (required, string): Completion type
Allowed values: "text", "stt", "tts"
params (required, object): Provider and type-specific parameters

params Fields (type: "text")
model (required, string): Model identifier (e.g., "gpt-4o", "gemini-2.5-pro")
instructions (optional, string): System prompt that guides model behavior and response style
knowledge_base_ids (optional, array of strings): Vector store IDs for knowledge retrieval (e.g., ["vs_123", "vs_456"])
reasoning (optional, string): Reasoning mode configuration
Allowed values: "low", "medium", "high"
temperature (optional, number): Sampling temperature controlling randomness (range: 0–2, default: 0.7)
max_tokens (optional, integer): Maximum tokens in the response (default: 1000)
max_num_results (optional, integer): Maximum number of results to return
provider_specific (optional, object): Provider-specific overrides and extensions
openai (optional, object): OpenAI-specific parameters
google (optional, object): Google-specific parameters

params Fields (type: "stt")
model (required, string): Speech-to-text model identifier (e.g., "whisper-1", "gemini-2.5-pro")
instruction (optional, string): Transcription instructions (e.g., "Transcribe the audio verbatim")
input_language (optional, string): Source audio language code (e.g., "hi" for Hindi)
output_language (optional, string): Target transcription language code (e.g., "en" for English)
When different from input_language, enables translation during transcription.
Default same as the input language
response_format (optional, string): Output format
Allowed values: "text", "json"
temperature (optional, number): Sampling temperature (range: 0–2, default: 0.7)
provider_specific (optional, object): Provider-specific overrides
openai (optional, object): OpenAI-specific parameters
google (optional, object): Google-specific parameters
params Fields (type: "tts")
model (required, string): Text-to-speech model identifier (e.g., "tts-1", "gemini-2.5-pro-tts")
voice (required, string): Voice identifier (e.g., "alloy", "nova")
language (required, string): Output audio language code (e.g., "en-US")
response_format (optional, string): Audio output format
Allowed values: "mp3", "wav", "ogg".
Default “wav”
speed (optional, number): Speech rate multiplier (range: 0.25–4.0, default: 1.0)
provider_specific (optional, object): Provider-specific overrides
openai (optional, object): OpenAI-specific parameters
google (optional, object): Google-specific parameters
director_notes (optional, string): Instructions for prosody and delivery style
response_modalities (optional, array of strings): Output modality types (e.g., ["AUDIO"])

The app/models/request.py and app/models/response.py have the pydantic models can be utilised for the llm_call table creation.
