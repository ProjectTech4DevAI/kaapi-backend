import logging
import os
import uuid
from typing import Any
from sarvamai import SarvamAI
from app.core.audio_utils import AudioRef
from app.models.llm import (
    NativeCompletionConfig,
    LLMCallResponse,
    QueryParams,
    TextOutput,
    LLMResponse,
    Usage,
    TextContent,
)
from app.models.llm.response import AudioOutput
from app.models.llm.request import AudioContent
from app.services.llm.providers.base import BaseProvider


logger = logging.getLogger(__name__)


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

    def _parse_input(
        self, query_input: Any, completion_type: str, provider: str
    ) -> str:
        if completion_type == "stt":
            if isinstance(query_input, str) and os.path.exists(query_input):
                return query_input
            else:
                raise ValueError(f"{provider} STT requires a valid file path as input")
        elif completion_type == "tts":
            if isinstance(query_input, str):
                return query_input
            else:
                raise ValueError(f"{provider} TTS requires a text string as input")
        raise ValueError(
            f"Unsupported completion type '{completion_type}' for {provider}"
        )

    def _execute_stt(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: "AudioRef",
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute speech-to-text completion using SarvamAI.

        Args:
            completion_config: Configuration for the completion request (with already-mapped params)
            resolved_input: ``AudioRef`` carrying the audio bytes; materialized to a temp file
                because the SarvamAI SDK only accepts a file-like.
            include_provider_raw_response: Whether to include raw provider response

        Returns:
            Tuple of (response, error_message)
        """
        provider_name = completion_config.provider
        params = completion_config.params

        if not isinstance(resolved_input, AudioRef):
            return None, f"{provider_name} STT requires AudioRef input"

        # Extract already-mapped parameters from the mapper
        model = params.get("model") or "saaras:v3"
        language_code = params.get("language_code")
        mode = params.get("mode") or "transcribe"

        try:
            with resolved_input.to_path() as parsed_input_path:
                stt_kwargs = {"file": None, "model": model}

                if language_code:
                    stt_kwargs["language_code"] = language_code
                if mode:
                    stt_kwargs["mode"] = mode

                with open(parsed_input_path, "rb") as audio_file:
                    stt_kwargs["file"] = audio_file
                    sarvam_response = self.client.speech_to_text.transcribe(
                        **stt_kwargs
                    )

            input_tokens_estimate = 0
            output_tokens_estimate = len(sarvam_response.transcript.split())
            total_tokens_estimate = input_tokens_estimate + output_tokens_estimate

            llm_response = LLMCallResponse(
                response=LLMResponse(
                    provider_response_id=sarvam_response.request_id
                    or str(uuid.uuid4()),
                    conversation_id=None,
                    provider=provider_name,
                    model=model,
                    output=TextOutput(
                        content=TextContent(value=sarvam_response.transcript)
                    ),
                ),
                usage=Usage(
                    input_tokens=input_tokens_estimate,
                    output_tokens=output_tokens_estimate,
                    total_tokens=total_tokens_estimate,
                    reasoning_tokens=None,
                ),
            )

            if include_provider_raw_response:
                llm_response.provider_raw_response = sarvam_response.model_dump()

            logger.info(
                f"[_execute_stt] Successfully transcribed audio | "
                f"request_id={sarvam_response.request_id}, provider={provider_name} model={model}, mode={mode}"
            )
            return llm_response, None

        except Exception as e:
            error_message = f"SarvamAI STT transcription failed: {str(e)}"
            logger.error(
                f"[_execute_stt] {error_message} | provider={provider_name}",
                exc_info=True,
            )
            return None, error_message

    def _execute_tts(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute text-to-speech completion using SarvamAI.

        Args:
            completion_config: Configuration for the completion request (with already-mapped params)
            resolved_input: Text string to convert to speech
            include_provider_raw_response: Whether to include raw provider response

        Returns:
            Tuple of (response, error_message)
        """
        provider_name = completion_config.provider
        params = completion_config.params

        # Extract already-mapped parameters from the mapper
        model = params.get("model") or "bulbul:v3"
        target_language_code = params.get("target_language_code")
        if not target_language_code:
            return (
                None,
                "Missing 'target_language_code' in native params for SarvamAI TTS",
            )

        # Optional parameters (have API defaults)
        speaker = params.get("speaker")  # Defaults: Shubh (v3) / Anushka (v2)
        output_audio_codec = params.get("output_audio_codec")  # Has API default

        # Parse and validate input
        parsed_text = self._parse_input(
            query_input=resolved_input,
            completion_type="tts",
            provider=provider_name,
        )

        try:
            # Build kwargs for API call, only including non-None parameters
            tts_kwargs = {
                "text": parsed_text,
                "target_language_code": target_language_code,
                "model": model,
            }

            if speaker:
                tts_kwargs["speaker"] = speaker

            if output_audio_codec:
                tts_kwargs["output_audio_codec"] = output_audio_codec

            # Call SarvamAI TTS with mapped parameters
            sarvam_response = self.client.text_to_speech.convert(**tts_kwargs)

            # SarvamAI returns a list of base64-encoded audio strings
            # For single text input, take the first audio
            if not sarvam_response.audios or len(sarvam_response.audios) == 0:
                return None, "SarvamAI TTS returned no audio data"

            audio_base64 = sarvam_response.audios[0]

            # Estimate token usage (not directly provided by SarvamAI TTS)
            input_tokens_estimate = len(parsed_text.split())
            output_tokens_estimate = 0  # Audio output, no tokens
            total_tokens_estimate = input_tokens_estimate

            llm_response = LLMCallResponse(
                response=LLMResponse(
                    provider_response_id=sarvam_response.request_id
                    or str(uuid.uuid4()),
                    conversation_id=None,
                    provider=provider_name,
                    model=model,
                    output=AudioOutput(
                        content=AudioContent(
                            format="base64",
                            value=audio_base64,
                            mime_type=f"audio/{output_audio_codec or 'wav'}",
                        )
                    ),
                ),
                usage=Usage(
                    input_tokens=input_tokens_estimate,
                    output_tokens=output_tokens_estimate,
                    total_tokens=total_tokens_estimate,
                    reasoning_tokens=None,
                ),
            )

            if include_provider_raw_response:
                llm_response.provider_raw_response = sarvam_response.model_dump()

            logger.info(
                f"[_execute_tts] Successfully converted text to speech | "
                f"request_id={sarvam_response.request_id}, provider={provider_name}, model={model}, speaker={speaker}"
            )
            return llm_response, None

        except Exception as e:
            error_message = f"SarvamAI TTS conversion failed: {str(e)}"
            logger.error(
                f"[_execute_tts] {error_message} | provider={provider_name}",
                exc_info=True,
            )
            return None, error_message

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,  # noqa: ARG002 - Required by base class interface, unused for STT/TTS
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        provider_name = completion_config.provider
        try:
            completion_type = completion_config.type

            if completion_type == "stt":
                return self._execute_stt(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            elif completion_type == "tts":
                return self._execute_tts(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            else:
                return (
                    None,
                    f"Unsupported completion type '{completion_type}' for SarvamAIProvider",
                )

        except ValueError as e:
            error_message = f"Input validation error: {str(e)}"
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | provider={provider_name}",
                exc_info=True,
            )
            return None, error_message
        except Exception as e:
            error_message = "Unexpected error occurred during SarvamAI execution"
            logger.error(
                f"[SarvamAIProvider.execute] {error_message}: {str(e)} | provider={provider_name}",
                exc_info=True,
            )
            return None, error_message
