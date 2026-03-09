import base64
import logging
import os
from typing import Any

from elevenlabs import ElevenLabs, SpeechToTextConvertResponse


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


class ElevenlabsAIProvider(BaseProvider):
    def __init__(self, client: ElevenLabs):
        """Initialize Elevenlabs provider with client.

        Args:
            client: ElevenLabs client instance
        """
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        if "api_key" not in credentials:
            raise ValueError("API Key for Elevenlabs Not Set")
        return ElevenLabs(api_key=credentials["api_key"])

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
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute speech-to-text completion using Elevenlabs.

        Args:
            completion_config: Configuration for the completion request (with already-mapped params)
            resolved_input: File path to the audio input
            include_provider_raw_response: Whether to include raw provider response

        Returns:
            Tuple of (response, error_message)
        """
        provider_name = completion_config.provider
        params = completion_config.params

        # Extract already-mapped parameters from the mapper
        model = params.get("model")
        if not model:
            return None, "Missing 'model' in native params for Elevenlabs STT"

        language_code = params.get("language_code")

        # Parse and validate input
        parsed_input_path = self._parse_input(
            query_input=resolved_input,
            completion_type="stt",
            provider=provider_name,
        )

        try:
            with open(parsed_input_path, "rb") as audio_file:
                # Call ElevenLabs transcribe with all mapped parameters
                elevenlabs_response: SpeechToTextConvertResponse = (
                    self.client.speech_to_text.convert(
                        file=audio_file, model_id=model, language_code=language_code
                    )
                )

            # Estimate token usage (not directly provided by Elevenlabs STT)
            input_tokens_estimate = 0
            output_tokens_estimate = len(elevenlabs_response.text.split())
            total_tokens_estimate = input_tokens_estimate + output_tokens_estimate

            llm_response = LLMCallResponse(
                response=LLMResponse(
                    provider_response_id=elevenlabs_response.transcription_id
                    or "unknown",
                    conversation_id=None,
                    provider=provider_name,
                    model=model,
                    output=TextOutput(
                        content=TextContent(value=elevenlabs_response.text)
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
                llm_response.provider_raw_response = elevenlabs_response.model_dump()

            logger.info(
                f"[_execute_stt] Successfully transcribed audio | "
                f"request_id={elevenlabs_response.transcription_id}, model={model}"
            )
            return llm_response, None

        except Exception as e:
            error_message = f"Elevenlabs STT transcription failed: {str(e)}"
            logger.error(f"[_execute_stt] {error_message}", exc_info=True)
            return None, error_message

    def _execute_tts(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute text-to-speech completion using Elevenlabs.

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
        model = params.get("model")
        if not model:
            return None, "Missing 'model' in native params for Elevenlabs TTS"

        voice_id = params.get("voice_id")
        if not voice_id:
            return None, "Missing 'voice_id' in native params for Elevenlabs TTS"

        output_format = params.get("output_format", "mp3_44100_128")
        language_code = params.get("language_code")
        voice_settings = params.get("voice_settings")

        # Parse and validate input
        parsed_text = self._parse_input(
            query_input=resolved_input,
            completion_type="tts",
            provider=provider_name,
        )

        try:
            # Build optional kwargs
            tts_kwargs: dict[str, Any] = {}
            if language_code:
                tts_kwargs["language_code"] = language_code
            if voice_settings:
                tts_kwargs["voice_settings"] = voice_settings

            # Call Elevenlabs TTS API
            audio_iterator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=parsed_text,
                model_id=model,
                output_format=output_format,
                **tts_kwargs,
            )

            # Elevenlabs returns an iterator of audio bytes; collect and base64-encode
            audio_bytes = b"".join(audio_iterator)
            if not audio_bytes:
                return None, "Elevenlabs TTS returned no audio data"

            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Derive mime type from output_format (e.g. "mp3_44100_128" -> "audio/mpeg")
            codec = output_format.split("_")[0]
            mime_type_map = {
                "mp3": "audio/mpeg",
                "pcm": "audio/pcm",
                "wav": "audio/wav",
                "opus": "audio/opus",
                "ulaw": "audio/basic",
                "alaw": "audio/alaw",
            }
            mime_type = mime_type_map.get(codec, f"audio/{codec}")

            # Estimate token usage (not directly provided by Elevenlabs TTS)
            input_tokens_estimate = len(parsed_text.split())
            output_tokens_estimate = 0  # Audio output, no tokens
            total_tokens_estimate = input_tokens_estimate

            llm_response = LLMCallResponse(
                response=LLMResponse(
                    provider_response_id="unknown",
                    conversation_id=None,
                    provider=provider_name,
                    model=model,
                    output=AudioOutput(
                        content=AudioContent(
                            format="base64",
                            value=audio_base64,
                            mime_type=mime_type,
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
                llm_response.provider_raw_response = {
                    "audio_bytes_length": len(audio_bytes),
                    "output_format": output_format,
                }

            logger.info(
                f"[_execute_tts] Successfully converted text to speech | "
                f"model={model}, voice_id={voice_id}, output_format={output_format}"
            )
            return llm_response, None

        except Exception as e:
            error_message = f"Elevenlabs TTS conversion failed: {str(e)}"
            logger.error(f"[_execute_tts] {error_message}", exc_info=True)
            return None, error_message

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,  # noqa: ARG002 - Required by base class interface, unused for STT/TTS
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
            elif completion_type == "tts":
                return self._execute_tts(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            else:
                return (
                    None,
                    f"Unsupported completion type '{completion_type}' for ElevenlabsAIProvider",
                )

        except ValueError as e:
            error_message = f"Input validation error: {str(e)}"
            logger.error(
                f"[ElevenlabsAIProvider.execute] {error_message}", exc_info=True
            )
            return None, error_message
        except Exception as e:
            error_message = "Unexpected error occurred during Elevenlabs execution"
            logger.error(
                f"[ElevenlabsAIProvider.execute] {error_message}: {str(e)}",
                exc_info=True,
            )
            return None, error_message
