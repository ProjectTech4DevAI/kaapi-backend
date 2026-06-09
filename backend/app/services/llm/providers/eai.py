import base64
import logging
import os
import uuid
from typing import Any

from elevenlabs import (
    ElevenLabs,
    errors as elevenlabs_errors,
    SpeechToTextConvertResponse,
)
from elevenlabs.core.api_error import ApiError as ElevenLabsApiError

# from elevenlabs.types import SpeechToTextConvertResponse
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
from app.models.llm.constants import (
    DEFAULT_ELEVENLABS_STT_MODEL,
    DEFAULT_ELEVENLABS_TTS_MODEL,
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
        resolved_input: "AudioRef",
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute speech-to-text completion using Elevenlabs.

        Args:
            completion_config: Configuration for the completion request (with already-mapped params)
            resolved_input: ``AudioRef``; materialized to a temp file because the
                ElevenLabs SDK only accepts a file-like.
            include_provider_raw_response: Whether to include raw provider response

        Returns:
            Tuple of (response, error_message)

        Note:
            SDK exceptions (ElevenLabs ApiError subclasses, TypeError, generic) are NOT
            caught here; they bubble up to ``execute()`` which handles them uniformly.
            This method only handles Kaapi-side input validation and ElevenLabs
            response-shape checks.
        """
        provider_name = completion_config.provider
        params = completion_config.params

        if not isinstance(resolved_input, AudioRef):
            error_message = (
                f"[KAAPI] STT validation failed: {provider_name} STT requires "
                f"an AudioRef input, but received {type(resolved_input).__name__}. "
                f"Ensure the audio is uploaded and resolved before invoking STT."
            )
            logger.warning(
                f"[ElevenlabsAIProvider._execute_stt] {error_message} | provider={provider_name}"
            )
            return None, error_message

        # Extract already-mapped parameters from the mapper
        model_id = params.get("model_id") or DEFAULT_ELEVENLABS_STT_MODEL
        if not model_id:
            error_message = (
                "[KAAPI] STT validation failed: 'model_id' is missing in "
                "native params for Elevenlabs STT. Set a model_id via the "
                "completion config before invoking STT."
            )
            logger.warning(
                f"[ElevenlabsAIProvider._execute_stt] {error_message} | provider={provider_name}"
            )
            return None, error_message

        language_code = params.get("language_code")
        temperature = params.get("temperature")

        stt_kwargs: dict[str, Any] = {}
        if language_code:
            stt_kwargs["language_code"] = language_code
        if temperature is not None:
            stt_kwargs["temperature"] = temperature

        with resolved_input.to_path() as parsed_input_path, open(
            parsed_input_path, "rb"
        ) as audio_file:
            elevenlabs_response: SpeechToTextConvertResponse = (
                self.client.speech_to_text.convert(
                    file=audio_file, model_id=model_id, **stt_kwargs
                )
            )

        if not elevenlabs_response.text:
            error_message = (
                "[ELEVENLABS] STT response is missing transcribed text. "
                "ElevenLabs returned an empty transcript — verify the audio "
                "is audible, in a supported language, and in a supported "
                "format, then retry. If the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[ElevenlabsAIProvider._execute_stt] {error_message} | "
                f"provider={provider_name}, model={model_id}, "
                f"transcription_id={elevenlabs_response.transcription_id}"
            )
            return None, error_message

        # Estimate token usage (not directly provided by Elevenlabs STT)
        input_tokens_estimate = 0
        output_tokens_estimate = len(elevenlabs_response.text.split())
        total_tokens_estimate = input_tokens_estimate + output_tokens_estimate
        transcription_id = elevenlabs_response.transcription_id or str(uuid.uuid4())
        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=transcription_id,
                conversation_id=None,
                provider=provider_name,
                model=model_id,
                output=TextOutput(content=TextContent(value=elevenlabs_response.text)),
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
            f"request_id={elevenlabs_response.transcription_id}, model={model_id}, provider={provider_name}"
        )
        return llm_response, None

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

        Note:
            SDK exceptions and ``_parse_input`` ValueErrors bubble up to ``execute()``.
        """
        provider_name = completion_config.provider
        params = completion_config.params

        # Extract already-mapped parameters from the mapper
        # Use 'or' to handle both missing keys and falsy values
        model_id = params.get("model_id") or DEFAULT_ELEVENLABS_TTS_MODEL
        voice_id = params.get("voice_id") or "EXAVITQu4vr4xnSDxMaL"

        if not model_id:
            error_message = (
                "[KAAPI] TTS validation failed: 'model_id' is missing in "
                "native params for Elevenlabs TTS. Set a model_id via the "
                "completion config before invoking TTS."
            )
            logger.warning(
                f"[ElevenlabsAIProvider._execute_tts] {error_message} | provider={provider_name}"
            )
            return None, error_message
        if not voice_id:
            error_message = (
                "[KAAPI] TTS validation failed: 'voice_id' is missing in "
                "native params for Elevenlabs TTS. Set a voice_id via the "
                "completion config before invoking TTS."
            )
            logger.warning(
                f"[ElevenlabsAIProvider._execute_tts] {error_message} | provider={provider_name}"
            )
            return None, error_message

        output_format = params.get("output_format", "wav_24000")
        language_code = params.get("language_code")
        voice_settings = params.get("voice_settings")

        # _parse_input may raise ValueError — let it bubble to execute()
        parsed_text = self._parse_input(
            query_input=resolved_input,
            completion_type="tts",
            provider=provider_name,
        )

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
            model_id=model_id,
            output_format=output_format,
            **tts_kwargs,
        )

        # Elevenlabs returns an iterator of audio bytes; collect and base64-encode
        audio_bytes = b"".join(audio_iterator)
        if not audio_bytes:
            error_message = (
                "[ELEVENLABS] TTS response contains no audio data. "
                "ElevenLabs accepted the request but returned an empty audio "
                "stream — this is typically an ElevenLabs server-side issue. "
                "Wait a minute and retry; if the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[ElevenlabsAIProvider._execute_tts] {error_message} | "
                f"provider={provider_name}, model={model_id}, voice_id={voice_id}"
            )
            return None, error_message

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
                provider_response_id=str(uuid.uuid4()),
                conversation_id=None,
                provider=provider_name,
                model=model_id,
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
            f"provider={provider_name}, model={model_id}, voice_id={voice_id}, output_format={output_format}"
        )
        return llm_response, None

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,  # noqa: ARG002 - Required by base class interface, unused for STT/TTS
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        provider_name = completion_config.provider
        completion_type = completion_config.type
        try:
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
                error_message = (
                    f"[KAAPI] Unsupported completion type '{completion_type}' "
                    f"for ElevenlabsAIProvider. ElevenLabs currently supports "
                    f"'stt' and 'tts' only."
                )
                logger.warning(
                    f"[ElevenlabsAIProvider.execute] {error_message} | provider={provider_name}"
                )
                return None, error_message

        except elevenlabs_errors.BadRequestError as e:
            error_message = (
                f"[ELEVENLABS] {completion_type.upper()} bad request "
                f"(code: 400): {e.body}. Review the request parameters — the "
                f"input file, language_code, model_id, voice_id, "
                f"output_format, or voice_settings may be invalid for this "
                f"ElevenLabs endpoint."
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except elevenlabs_errors.UnauthorizedError as e:
            error_message = (
                f"[ELEVENLABS] {completion_type.upper()} authentication "
                f"failed (code: 401): {e.body}. Verify the ElevenLabs API "
                f"key is valid, not expired, and has been correctly "
                f"configured for this project."
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except elevenlabs_errors.ForbiddenError as e:
            error_message = (
                f"[ELEVENLABS] {completion_type.upper()} permission denied "
                f"(code: 403): {e.body}. The API key does not have access to "
                f"the requested model, voice, or feature — check your "
                f"ElevenLabs plan and key permissions."
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except elevenlabs_errors.NotFoundError as e:
            error_message = (
                f"[ELEVENLABS] {completion_type.upper()} resource not found "
                f"(code: 404): {e.body}. Verify the model_id and voice_id "
                f"are correct and available on your ElevenLabs plan."
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except elevenlabs_errors.ConflictError as e:
            error_message = (
                f"[ELEVENLABS] {completion_type.upper()} conflict (code: 409): "
                f"{e.body}. The request conflicts with the current resource "
                f"state — review concurrent requests or resource locks before "
                f"retrying."
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except elevenlabs_errors.UnprocessableEntityError as e:
            error_message = (
                f"[ELEVENLABS] {completion_type.upper()} unprocessable entity "
                f"(code: 422): {e.body}. ElevenLabs rejected the request "
                f"payload — check text formatting, voice_settings, and "
                f"output_format values against the API spec."
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except elevenlabs_errors.TooEarlyError as e:
            error_message = (
                f"[ELEVENLABS] {completion_type.upper()} too early "
                f"(code: 425): {e.body}. ElevenLabs is not yet ready to "
                f"process this request — wait a few seconds and retry. If "
                f"the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except ElevenLabsApiError as e:
            # Catch-all for any ElevenLabs ApiError subclass not matched above.
            # This includes 429 (rate limit), 5xx (server errors), and any
            # other status code that doesn't have a dedicated subclass.
            status = e.status_code
            if status == 429:
                hint = (
                    "You have hit ElevenLabs' request rate or quota — wait "
                    "at least 1 minute and retry. If the issue persists, "
                    "request a quota increase from ElevenLabs or contact "
                    "Kaapi."
                )
            elif status is not None and 500 <= status < 600:
                hint = (
                    "ElevenLabs is experiencing a server-side issue — this "
                    "is typically transient. Retry in a few seconds; if the "
                    "issue persists, contact Kaapi."
                )
            else:
                hint = "If this persists, contact Kaapi."
            error_message = (
                f"[ELEVENLABS] {completion_type.upper()} API error "
                f"(code: {status}): {e.body}. {hint}"
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except ValueError as e:
            error_message = (
                f"[KAAPI] Input validation error during ElevenLabs "
                f"{completion_type or 'execution'}: {str(e)}. Review the "
                f"request input and config — one of the required fields is "
                f"missing or malformed."
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except TypeError as e:
            error_message = (
                f"[KAAPI] Invalid or unexpected parameter in Config: "
                f"{str(e)}. Review the completion config; one of the "
                f"parameters does not match ElevenLabs' expected signature."
            )
            logger.warning(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except Exception as e:
            error_message = (
                f"[KAAPI] Unexpected error during ElevenLabs "
                f"{completion_type or 'execution'}: {str(e)}. This was not "
                f"raised by the ElevenLabs SDK directly — likely a Kaapi-side "
                f"failure. Contact Kaapi if the issue persists."
            )
            logger.error(
                f"[ElevenlabsAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message
