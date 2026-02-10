import logging
import os
from typing import Any

from sarvamai import SarvamAI   



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

# SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
#if not SARVAM_API_KEY:
 #   SARVAM_API_KEY = "sk_lmsvfc31_On1bxqwDAqYZoijqBfblr3yf"  # for testing only
  #  print("Requested Action: Please set SARVAM_API_KEY , Going ahead with a trail key for testing purposes.")
 



class SarvamAIProvider(BaseProvider):
    def __init__(self, client: SarvamAI):
        """Initialize SarvamAI provider with client.

        Args:
            client: SarvamAI client instance
        """
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        if "api_key" not in credentials:
            raise ValueError("API Key for SarvamAI Not Set")
        return SarvamAI(api_subscription_key=credentials["api_key"])

    def _parse_input(self, query_input: Any, completion_type: str, provider: str) -> str:
        if completion_type == "stt":
            if isinstance(query_input, str) and os.path.exists(query_input):
                return query_input
            else:
                raise ValueError(f"{provider} STT requires a valid file path as input")
        raise ValueError(f"Unsupported completion type '{completion_type}' for {provider}")

    def _execute_stt(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute speech-to-text completion using SarvamAI.

        Args:
            completion_config: Configuration for the completion request
            resolved_input: File path to the audio input
            include_provider_raw_response: Whether to include raw provider response

        Returns:
            Tuple of (response, error_message)
        """
        provider_name = self.get_provider_name()
        generation_params = completion_config.params

        model = generation_params.get("model")
        if not model:
            return None, "Missing 'model' in native params for SarvamAI STT"
        
        inputlanguageofaudio = generation_params.get("input_language")
        if not inputlanguageofaudio:
          inputlanguageofaudio = "unknown" #'unknown' for automatic language detection or ISO 639 language code like 'hi-IN'. SarvamAI's Saarika model supports mixed language content with automatic detection of languages within the sentence, so this parameter is optional and can be set to "unknown" if not provided. 

        # Parse and validate input
        parsed_input_path = self._parse_input(
            query_input=resolved_input,
            completion_type="stt",
            provider=provider_name,
        )

        try:
            with open(parsed_input_path, "rb") as audio_file:
                sarvam_response = self.client.speech_to_text.transcribe(
                    file=audio_file,
                    model=model,
                    # SarvamAI's flagship STT model  Saarika supports mixed language content with automatic detection of languages within the sentance    
                    language_code=inputlanguageofaudio, # Optional, can be set to "unknown" for automatic detection or specific ISO 639 language code like 'hi-IN'
                )

            # SarvamAI does not provide token usage directly for STT, so we'll use placeholders
            # You might estimate based on transcript length or set to 0
            input_tokens_estimate = 0 # Not directly provided by SarvamAI STT
            output_tokens_estimate = len(sarvam_response.transcript.split()) # Estimate by word count
            total_tokens_estimate = input_tokens_estimate + output_tokens_estimate

            llm_response = LLMCallResponse(
                response=LLMResponse(
                    provider_response_id=sarvam_response.request_id or "unknown",
                    conversation_id=None,  # SarvamAI STT doesn't have conversation_id
                    provider=provider_name,
                    model=model,
                    output=LLMOutput(text=sarvam_response.transcript or ""),
                ),
                usage=Usage(
                    input_tokens=input_tokens_estimate,
                    output_tokens=output_tokens_estimate,
                    total_tokens=total_tokens_estimate,
                    reasoning_tokens=None, # Not provided by SarvamAI
                ),
            )

            if include_provider_raw_response:
                llm_response.provider_raw_response = sarvam_response.model_dump()

            logger.info(
                f"[{provider_name}.execute_stt] Successfully transcribed audio: {sarvam_response.request_id}"
            )
            return llm_response, None

        except Exception as e:
            error_message = f"SarvamAI STT transcription failed: {str(e)}"
            logger.error(f"[{provider_name}.execute_stt] {error_message}", exc_info=True)
            return None, error_message

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        try:
            completion_type = completion_config.type

            if completion_type == "stt":
                return self._execute_stt(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            else:
                return None, f"Unsupported completion type '{completion_type}' for SarvamAIProvider"

        except ValueError as e:
            error_message = f"Input validation error: {str(e)}"
            logger.error(f"[SarvamAIProvider.execute] {error_message}", exc_info=True)
            return None, error_message
        except Exception as e:
            error_message = "Unexpected error occurred during SarvamAI execution"
            logger.error(f"[SarvamAIProvider.execute] {error_message}: {str(e)}", exc_info=True)
            return None, error_message

