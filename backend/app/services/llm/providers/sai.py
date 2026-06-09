import logging
import os
import uuid
from typing import Any
from sarvamai import (
    SarvamAI,
    errors as sarvam_errors,
)
from sarvamai.core.api_error import ApiError as SarvamApiError

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
    DEFAULT_SARVAM_STT_MODEL,
    DEFAULT_SARVAM_TTS_MODEL,
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

        Note:
            SDK exceptions (Sarvam ApiError subclasses, TypeError, generic) are NOT caught
            here; they bubble up to ``execute()`` which handles them uniformly. This method
            only handles Kaapi-side input validation and Sarvam response-shape checks.
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
                f"[SarvamAIProvider._execute_stt] {error_message} | provider={provider_name}"
            )
            return None, error_message

        # Extract already-mapped parameters from the mapper
        model = params.get("model") or DEFAULT_SARVAM_STT_MODEL
        language_code = params.get("language_code")
        mode = params.get("mode") or "transcribe"

        with resolved_input.to_path() as parsed_input_path:
            stt_kwargs = {"file": None, "model": model}

            if language_code:
                stt_kwargs["language_code"] = language_code
            if mode:
                stt_kwargs["mode"] = mode

            with open(parsed_input_path, "rb") as audio_file:
                stt_kwargs["file"] = audio_file
                sarvam_response = self.client.speech_to_text.transcribe(**stt_kwargs)

        if not sarvam_response.transcript:
            error_message = (
                "[SARVAM] STT response is missing transcribed text. Sarvam "
                "returned an empty transcript — verify the audio is "
                "audible, in a supported language, and in a supported "
                "format, then retry. If the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[SarvamAIProvider._execute_stt] {error_message} | "
                f"provider={provider_name}, model={model}, "
                f"request_id={sarvam_response.request_id}"
            )
            return None, error_message

        input_tokens_estimate = 0
        output_tokens_estimate = len(sarvam_response.transcript.split())
        total_tokens_estimate = input_tokens_estimate + output_tokens_estimate

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=sarvam_response.request_id or str(uuid.uuid4()),
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

        Note:
            SDK exceptions (Sarvam ApiError subclasses, ValueError from _parse_input,
            TypeError, generic) are NOT caught here; they bubble up to ``execute()``.
        """
        provider_name = completion_config.provider
        params = completion_config.params

        # Extract already-mapped parameters from the mapper
        model = params.get("model") or DEFAULT_SARVAM_TTS_MODEL
        target_language_code = params.get("target_language_code")
        if not target_language_code:
            error_message = (
                "[KAAPI] TTS validation failed: 'target_language_code' is "
                "missing in native params for SarvamAI TTS. Sarvam requires "
                "an explicit target language — set it via the completion "
                "config before invoking TTS."
            )
            logger.warning(
                f"[SarvamAIProvider._execute_tts] {error_message} | provider={provider_name}"
            )
            return None, error_message

        # Optional parameters (have API defaults)
        speaker = params.get("speaker")  # Defaults: Shubh (v3) / Anushka (v2)
        output_audio_codec = params.get("output_audio_codec")  # Has API default

        # _parse_input may raise ValueError — let it bubble to execute()
        parsed_text = self._parse_input(
            query_input=resolved_input,
            completion_type="tts",
            provider=provider_name,
        )

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
            error_message = (
                "[SARVAM] TTS response contains no audio data. Sarvam "
                "accepted the request but returned an empty audio list — "
                "this is typically a Sarvam server-side issue. Wait a "
                "minute and retry; if the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[SarvamAIProvider._execute_tts] {error_message} | "
                f"provider={provider_name}, model={model}, "
                f"request_id={sarvam_response.request_id}"
            )
            return None, error_message

        audio_base64 = sarvam_response.audios[0]

        # Estimate token usage (not directly provided by SarvamAI TTS)
        input_tokens_estimate = len(parsed_text.split())
        output_tokens_estimate = 0  # Audio output, no tokens
        total_tokens_estimate = input_tokens_estimate

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=sarvam_response.request_id or str(uuid.uuid4()),
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
                    f"for SarvamAIProvider. Sarvam currently supports 'stt' "
                    f"and 'tts' only."
                )
                logger.warning(
                    f"[SarvamAIProvider.execute] {error_message} | provider={provider_name}"
                )
                return None, error_message

        except sarvam_errors.BadRequestError as e:
            error_message = (
                f"[SARVAM] {completion_type.upper()} bad request (code: 400): "
                f"{e.body}. Review the request parameters — the input file, "
                f"language_code, mode, speaker, or output_audio_codec may be "
                f"invalid for this Sarvam endpoint."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except sarvam_errors.ForbiddenError as e:
            error_message = (
                f"[SARVAM] {completion_type.upper()} authentication / permission "
                f"denied (code: 403): {e.body}. Verify the Sarvam API key is "
                f"valid, not expired, and has access to the requested model "
                f"and speaker."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except sarvam_errors.NotFoundError as e:
            error_message = (
                f"[SARVAM] {completion_type.upper()} resource not found "
                f"(code: 404): {e.body}. Verify the model name and any "
                f"referenced IDs in your config are correct and available "
                f"on your Sarvam plan."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except sarvam_errors.ContentTooLargeError as e:
            error_message = (
                f"[SARVAM] {completion_type.upper()} payload too large "
                f"(code: 413): {e.body}. The input (audio file for STT, text "
                f"for TTS) exceeds Sarvam's size limit — split into smaller "
                f"chunks or compress before retrying."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except sarvam_errors.UnprocessableEntityError as e:
            error_message = (
                f"[SARVAM] {completion_type.upper()} unprocessable entity "
                f"(code: 422): {e.body}. Sarvam rejected the request payload "
                f"— check input format and parameter values against the API spec."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except sarvam_errors.TooManyRequestsError as e:
            error_message = (
                f"[SARVAM] {completion_type.upper()} rate limit / quota "
                f"exceeded (code: 429): {e.body}. You have hit Sarvam's "
                f"request rate or quota — wait at least 1 minute and retry. "
                f"If the issue persists, request a quota increase from Sarvam "
                f"or contact Kaapi."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except sarvam_errors.InternalServerError as e:
            error_message = (
                f"[SARVAM] {completion_type.upper()} server error (code: 500): "
                f"{e.body}. This is typically transient — retry in a few "
                f"seconds. If the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except sarvam_errors.ServiceUnavailableError as e:
            error_message = (
                f"[SARVAM] {completion_type.upper()} service unavailable "
                f"(code: 503): {e.body}. Sarvam is overloaded or temporarily "
                f"down — retry in a few seconds. If the issue persists, "
                f"contact Kaapi."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except SarvamApiError as e:
            # Catch-all for any Sarvam ApiError subclass not matched above
            error_message = (
                f"[SARVAM] {completion_type.upper()} API error "
                f"(code: {e.status_code}): {e.body}. If this persists, "
                f"contact Kaapi."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except ValueError as e:
            error_message = (
                f"[KAAPI] Input validation error during Sarvam "
                f"{completion_type or 'execution'}: {str(e)}. Review the "
                f"request input and config — one of the required fields is "
                f"missing or malformed."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except TypeError as e:
            error_message = (
                f"[KAAPI] Invalid or unexpected parameter in Config: {str(e)}. "
                f"Review the completion config; one of the parameters does "
                f"not match Sarvam's expected signature."
            )
            logger.warning(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except Exception as e:
            error_message = (
                f"[KAAPI] Unexpected error during Sarvam "
                f"{completion_type or 'execution'}: {str(e)}. This was not "
                f"raised by the Sarvam SDK directly — likely a Kaapi-side "
                f"failure. Contact Kaapi if the issue persists."
            )
            logger.error(
                f"[SarvamAIProvider.execute] {error_message} | "
                f"provider={provider_name}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message
