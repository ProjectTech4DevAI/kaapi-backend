import logging
import os
from typing import Any

import io
import base64
from elevenlabs.client import ElevenLabs
from elevenlabs import speech_to_text
from typing import Any, Dict, Optional, Tuple


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




class BaseProvider:
    """Base class for all LLM providers to ensure common interface."""
    pass

class NativeCompletionConfig:
    """Configuration for native LLM completions."""
    def __init__(self, provider: str, type: str, params: Optional[Dict[str, Any]] = None):
        self.provider = provider
        self.type = type
        self.params = params if params is not None else {}

class LLMCallResponse:
    """Standardized response object for LLM calls."""
    def __init__(self, text: Optional[str] = None, model: Optional[str] = None,
                 usage: Optional[Dict[str, Any]] = None,
                 provider_raw_response: Optional[Any] = None,
                 error: Optional[str] = None):
        self.text = text
        self.model = model
        self.usage = usage if usage is not None else {}
        self.provider_raw_response = provider_raw_response
        self.error = error

class QueryParams:
    """Placeholder for query parameters."""
    def __init__(self, input: Dict[str, Any]):
        self.input = input


class ElevenLabsAIProvider(BaseProvider):
    """Provider for ElevenLabs AI services, specifically Speech-to-Text."""

    def __init__(self, client: ElevenLabs):
        self.client = client

    @staticmethod
    def create_client(credentials: Dict[str, Any]) -> ElevenLabs:
        """Creates an ElevenLabs client from credentials."""
        api_key = credentials.get('api_key')
        if not api_key:
            raise ValueError("ElevenLabs API key is missing in credentials.")
        return ElevenLabs(api_key=api_key)

    def _parse_input(self, query_input: Any, completion_type: str, provider: str) -> Any:
        """Parses and validates input for ElevenLabs STT. Returns the input as is for STT."""
        # ElevenLabs speech_to_text.convert can take file path or BytesIO, so we pass it through
        if not isinstance(query_input, (str, io.BytesIO, io.BufferedReader)): # BufferedReader for opened files
            raise TypeError(f"query_input for {provider} {completion_type} must be a file path string or file-like object. Got {type(query_input)}")
        return query_input

    def _execute_stt(self, completion_config: NativeCompletionConfig, resolved_input: Any, include_provider_raw_response: bool = False) -> Tuple[Optional[LLMCallResponse], Optional[str]]:
        """Executes Speech-to-Text conversion using ElevenLabs."""
        try:
            audio_file_to_send = resolved_input

            # Ensure BytesIO is at the beginning if passed directly
            if isinstance(audio_file_to_send, io.BytesIO):
                audio_file_to_send.seek(0)

            # Extract parameters from completion_config.params
            stt_params = completion_config.params
            model_name = stt_params.get("model_name", "scribe_v2")
            input_language = stt_params.get("input_language")
            diarize = stt_params.get("diarize", False)
            tag_audio_events = stt_params.get("tag_audio_events", False)

            # Perform STT conversion
            stt_response = self.client.speech_to_text.convert(
                file=audio_file_to_send,
                model_id=model_name,
                language_code=input_language,
                diarize=diarize,
                tag_audio_events=tag_audio_events
            )

            transcribed_text = stt_response.text

            # ElevenLabs response doesn't provide token usage directly, use word count as an alternative metric
            usage = {
                'word_count': len(stt_response.words) if stt_response.words else 0,
                'input_type': 'audio'
            }

            llm_response = LLMCallResponse(
                text=transcribed_text,
                model=model_name,
                usage=usage,
                provider_raw_response=stt_response if include_provider_raw_response else None
            )
            return llm_response, None
        except Exception as e:
            return None, str(e)

    def execute(self, completion_config: NativeCompletionConfig, query: QueryParams, resolved_input: Any, include_provider_raw_response: bool = False) -> Tuple[Optional[LLMCallResponse], Optional[str]]:
        """Main execution method, delegates to specific completion types."""
        # Use completion_config.type instead of completion_config.completion_type
        if completion_config.type == "stt":
            # The _parse_input method was called before execute in GoogleAIProvider,
            # but resolved_input is already in a usable format here (file path or BytesIO).
            # So we can pass it directly.
            return self._execute_stt(completion_config, resolved_input, include_provider_raw_response)
        else:
            return None, f"Unsupported completion type for ElevenLabsAIProvider: {completion_config.type}"
