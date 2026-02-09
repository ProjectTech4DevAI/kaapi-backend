from google.cloud import speech_v2
from google.cloud.speech_v2 import types
import google.auth

import base64
from typing import Any, Dict, Optional, Tuple

import logging
import os
from typing import Any, Dict, Optional, Tuple, Literal, Union

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

# Google Speech-to-Text provider implementation using Google Cloud's Speech-to-Text API (speech_v2)
class GoogleSpeechtoTextAIProvider(BaseProvider):
    def __init__(self, client: speech_v2.SpeechClient, project_id: str):
        super().__init__(client)
        self.client = client
        self.project_id = project_id  # Store project_id for use in requests

    @staticmethod
    def create_client(credentials: Dict[str, Any]) -> speech_v2.SpeechClient:
        """Create and return a Google Cloud Speech-to-Text client instance.

        Args:
            credentials: A dictionary containing 'key_filename', 'project_id', and optionally 'api_endpoint'.

        Returns:
            A configured speech_v2.SpeechClient instance.
        """
        key_filename = credentials.get("key_filename")
        project_id = credentials.get("project_id")
        api_endpoint = credentials.get("api_endpoint", "global") # Default to global if not specified

        if not key_filename:
            raise ValueError("Service account key filename is not provided in credentials.")
        if not project_id:
            raise ValueError("Project ID is not provided in credentials.")

        creds, _ = google.auth.load_credentials_from_file(key_filename)

        client_options = {}
        if api_endpoint != "global":
            client_options["api_endpoint"] = f"{api_endpoint}-speech.googleapis.com"

        return speech_v2.SpeechClient(credentials=creds, client_options=client_options)

    def _parse_input(
        self,
        query_input: Any, # Can be str (file path) or bytes (raw audio)
        completion_type: str,
        provider: str
    ) -> str: # Returns a file path string, consistent with GoogleAIProvider's _parse_input
        """Parses and validates the query input. For STT, it ensures it's a valid file path.

        Args:
            query_input: The raw query input.
            completion_type: The type of completion (e.g., 'stt').
            provider: The name of the provider.

        Returns:
            A string representing the resolved input (e.g., file path).

        Raises:
            ValueError: If the input is not valid for the completion type.
        """
        if completion_type == "stt":
            if isinstance(query_input, str):
                if not os.path.exists(query_input):
                    raise ValueError(f"File not found at path: {query_input}")
                return query_input  # Return the file path string
            else:
                raise ValueError(f"{provider} STT requires a file path (str) as input.")
        # Add other completion types if needed, ensuring they return a string
        raise ValueError(f"Unsupported completion type: {completion_type} for {provider}")

    def _execute_stt(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str,  # Now expecting a file path (str), consistent with GoogleAIProvider
        include_provider_raw_response: bool = False,
    ) -> Tuple[Optional[LLMCallResponse], Optional[str]]:
        provider_name = self.get_provider_name()
        generation_params = completion_config.params

        model = generation_params.get("model", "chirp_3")
        language_codes = generation_params.get("language_codes", ["en-US"])
        recognizer_id = generation_params.get("recognizer_id", "_")
        location = generation_params.get("location", "global")

        current_project_id = self.project_id

        if not current_project_id:
             return None, "Missing project_id in GoogleSpeechtoTextAIProvider initialization."

        # Read the audio file into bytes within _execute_stt
        try:
            if not os.path.exists(resolved_input):
                return None, f"Audio file not found at path: {resolved_input}"
            with open(resolved_input, "rb") as f:
                resolved_input_bytes = f.read()
        except Exception as e:
            return None, f"Failed to read audio file: {str(e)}"

        config = types.RecognitionConfig(
            auto_decoding_config=types.AutoDetectDecodingConfig(),
            language_codes=language_codes,
            model=model,
        )

        try:
            request = types.RecognizeRequest(
                recognizer=f"projects/{current_project_id}/locations/{location}/recognizers/{recognizer_id}",
                config=config,
                content=resolved_input_bytes, # Direct content is passed here
            )

            response: types.RecognizeResponse = self.client.recognize(request=request)

            transcript = ""
            total_output_chars = 0
            if response.results:
                for result in response.results:
                    if result.alternatives:
                        transcript += result.alternatives[0].transcript
                        total_output_chars += len(result.alternatives[0].transcript)

            input_tokens_estimate = 0
            output_tokens_estimate = total_output_chars
            total_tokens_estimate = input_tokens_estimate + output_tokens_estimate

            llm_response = LLMCallResponse(
                response=LLMResponse(
                    provider_response_id=response.request_id if hasattr(response, "request_id") else "unknown",
                    conversation_id=None,
                    provider=provider_name,
                    model=model,
                    output=LLMOutput(text=transcript),
                ),
                usage=Usage(
                    input_tokens=input_tokens_estimate,
                    output_tokens=output_tokens_estimate,
                    total_tokens=total_tokens_estimate,
                    reasoning_tokens=None,
                ),
            )

            if include_provider_raw_response:
                llm_response.provider_raw_response = types.RecognizeResponse.to_dict(response)

            logger.info(
                f"[{provider_name}.execute_stt] Successfully transcribed audio: {llm_response.response.provider_response_id}"
            )
            return llm_response, None

        except Exception as e:
            error_message = f"Google Cloud Speech-to-Text transcription failed: {str(e)}"
            logger.error(f"[{provider_name}.execute_stt] {error_message}", exc_info=True)
            return None, error_message

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str,  # This is expected to be a file path for STT
        include_provider_raw_response: bool = False,
    ) -> Tuple[Optional[LLMCallResponse], Optional[str]]:
        try:
            completion_type = completion_config.type

            if completion_type == "stt":
                # _parse_input now returns the validated file path string
                validated_file_path = self._parse_input(
                    query_input=resolved_input,
                    completion_type="stt",
                    provider=self.get_provider_name(),
                )

                return self._execute_stt(
                    completion_config=completion_config,
                    resolved_input=validated_file_path, # Pass the file path string
                    include_provider_raw_response=include_provider_raw_response,
                )
            else:
                return (
                    None,
                    f"Unsupported completion type '{completion_type}' for {self.get_provider_name()}",
                )

        except ValueError as e:
            error_message = f"Input validation error: {str(e)}"
            logger.error(
                f"[{self.get_provider_name()}.execute] {error_message}", exc_info=True
            )
            return None, error_message
        except Exception as e:
            error_message = f"Unexpected error occurred during {self.get_provider_name()} execution: {str(e)}"
            logger.error(
                f"[{self.get_provider_name()}.execute] {error_message}", exc_info=True
            )
            return None, error_message