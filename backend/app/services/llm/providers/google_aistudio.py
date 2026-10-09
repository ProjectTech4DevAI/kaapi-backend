import base64
import io
import logging
import wave
from typing import Any, cast

from google import genai
from google.genai import errors as genai_errors

# Interactions raises its own error hierarchy, not google.genai.errors; in
# google-genai 2.29 this private module is the only path that exports it.
from google.genai._gaos.lib import compat_errors as interactions_errors
from google.genai.interactions import Interaction

from app.core.audio_utils import (
    AudioRef,
    convert_pcm_to_mp3,
    convert_pcm_to_ogg,
    pcm_to_wav,
)
from app.models.llm import (
    ImageContent,
    LLMCallResponse,
    LLMResponse,
    NativeCompletionConfig,
    PDFContent,
    QueryParams,
    TextContent,
    TextOutput,
    Usage,
)
from app.models.llm.constants import (
    DEFAULT_STT_MODEL,
    DEFAULT_TEXT_MODELS,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    CompletionType,
)
from app.models.llm.response import AudioContent, AudioOutput
from app.services.llm.providers.base import BaseProvider, ContentPart, MultiModalInput

logger = logging.getLogger(__name__)


class GoogleAIProvider(BaseProvider):
    def __init__(self, client: genai.Client):
        """Initialize Google AI provider with client.

        Args:
            client: Google AI client instance
        """
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        if "api_key" not in credentials:
            raise ValueError("API Key for Google Gemini Not Set")
        return genai.Client(api_key=credentials["api_key"])

    @staticmethod
    def format_parts(
        parts: list[ContentPart],
    ) -> list[dict[str, Any]]:
        """Render Kaapi content parts as Interactions content items."""
        items: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextContent):
                items.append({"type": "text", "text": part.value})
                continue

            if isinstance(part, ImageContent):
                item: dict[str, Any] = {"type": "image"}
            elif isinstance(part, PDFContent):
                item = {"type": "document"}
            else:
                continue

            if part.format == "base64":
                item["data"] = part.value
            else:
                item["uri"] = part.value
            if part.mime_type:
                item["mime_type"] = part.mime_type
            items.append(item)
        return items

    @staticmethod
    def _extract_usage(interaction: Interaction, provider: str) -> Usage:
        usage = interaction.usage
        if usage is None:
            logger.warning(
                f"[GoogleAIProvider._extract_usage] Interaction missing usage, using zeros | "
                f"provider={provider}, interaction_id={interaction.id}"
            )
            return Usage(
                input_tokens=0, output_tokens=0, total_tokens=0, reasoning_tokens=0
            )

        return Usage(
            input_tokens=usage.total_input_tokens or 0,
            output_tokens=usage.total_output_tokens or 0,
            total_tokens=usage.total_tokens or 0,
            reasoning_tokens=usage.total_thought_tokens or 0,
        )

    def _execute_stt(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: "AudioRef",
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute speech-to-text completion using Google AI.

        Args:
            completion_config: Configuration for the completion request
            resolved_input: ``AudioRef``; materialized to a temp file because the
                google-genai SDK's ``files.upload`` expects a filesystem path.
            include_provider_raw_response: Whether to include raw provider response

        Returns:
            Tuple of (LLMCallResponse, error_message)
        """
        provider = completion_config.provider
        generation_params = completion_config.params

        if not isinstance(resolved_input, AudioRef):
            error_message = (
                f"[KAAPI] STT validation failed: {provider} STT requires an "
                f"AudioRef input, but received {type(resolved_input).__name__}. "
                f"Ensure the audio is uploaded and resolved before invoking STT."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_stt] {error_message} | provider={provider}"
            )
            return None, error_message

        model = generation_params.get("model") or DEFAULT_STT_MODEL
        instructions = generation_params.get("instructions", "")
        input_language = generation_params.get("input_language") or "auto"
        output_language = generation_params.get("output_language", "")
        thinking_level = generation_params.get("thinking_level") or "low"

        if input_language == "auto":
            lang_instruction = (
                "Detect the spoken language automatically and transcribe the audio"
            )
        else:
            lang_instruction = f"Transcribe the audio from {input_language} in the native script of {input_language}"

        if output_language and output_language != input_language:
            lang_instruction += f" and translate to {output_language} in the native script of {output_language} and only return transcribed script in {output_language}."

        forced_transcription_text = "Only return transcribed text and no other text."
        if instructions:
            merged_instruction = (
                f"{instructions}. {lang_instruction}. {forced_transcription_text}"
            )
        else:
            merged_instruction = f"{lang_instruction}. {forced_transcription_text}"

        logger.info(
            f"[GoogleAIProvider._execute_stt] Built transcription prompt | "
            f"provider={provider}, model={model}, input_language={input_language}, "
            f"output_language={output_language}"
        )

        with resolved_input.to_path() as audio_path:
            gemini_file = self.client.files.upload(file=audio_path)

        interaction = cast(
            Interaction,
            self.client.interactions.create(
                model=model,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {"type": "text", "text": merged_instruction},
                            {
                                "type": "audio",
                                "uri": gemini_file.uri,
                                "mime_type": gemini_file.mime_type,
                            },
                        ],
                    }
                ],
                generation_config={"thinking_level": thinking_level},
            ),
        )

        response_id = interaction.id
        if not response_id or interaction.status != "completed":
            error_message = (
                f"[GEMINI] STT interaction did not complete (status="
                f"{interaction.status}, errors={interaction.errors}). Retry the "
                f"request; if the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_stt] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        if not interaction.output_text:
            error_message = (
                "[GEMINI] STT response is missing transcribed text. Gemini "
                "returned an empty result — verify the audio is audible and in "
                "a supported format, then retry. If the issue persists, "
                "contact Kaapi."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_stt] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response_id,
                model=interaction.model or model,
                provider=provider,
                output=TextOutput(
                    content=TextContent(
                        value=interaction.output_text, language_code=output_language
                    )
                ),
            ),
            usage=self._extract_usage(interaction, provider),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = interaction.model_dump(mode="json")

        logger.info(
            f"[GoogleAIProvider._execute_stt] Successfully generated STT response | "
            f"request_id={response_id}, provider={provider}, model={model}"
        )

        return llm_response, None

    def _execute_tts(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute text-to-speech completion using Google AI.

        Args:
            completion_config: Configuration for the completion request
            resolved_input: Text string to synthesize
            include_provider_raw_response: Whether to include raw provider response

        Returns:
            Tuple of (LLMCallResponse, error_message)
        """
        provider = completion_config.provider
        generation_params = completion_config.params

        if not isinstance(resolved_input, str):
            error_message = (
                f"[KAAPI] TTS validation failed: {provider} TTS requires a text "
                f"string as input, but received {type(resolved_input).__name__}. "
                f"Provide the text to synthesize as a plain string."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_tts] {error_message} | provider={provider}"
            )
            return None, error_message

        if not resolved_input.strip():
            error_message = (
                "[KAAPI] TTS validation failed: text input is empty or "
                "whitespace-only. Provide non-empty text to synthesize."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_tts] {error_message} | provider={provider}"
            )
            return None, error_message

        model = generation_params.get("model") or DEFAULT_TTS_MODEL
        voice = generation_params.get("voice") or DEFAULT_TTS_VOICE
        # Optional: Gemini auto-detects the language from the script when unset.
        language = generation_params.get("language")
        response_format = generation_params.get("response_format", "wav")

        provider_specific = generation_params.get("provider_specific", {})
        gemini_params = provider_specific.get("gemini", {})
        director_notes = gemini_params.get("director_notes", "")

        transcript: dict[str, Any] = {
            "type": "text",
            "text": f"<transcript>{resolved_input}</transcript>",
        }
        if director_notes:
            transcript["annotations"] = [
                {"type": "speech_metadata", "style": director_notes}
            ]

        speech_config: dict[str, Any] = {"voice": voice}
        if language:
            speech_config["language"] = language

        interaction = cast(
            Interaction,
            self.client.interactions.create(
                model=model,
                input=[{"type": "user_input", "content": [transcript]}],
                generation_config={"speech_config": [speech_config]},
                response_format={"type": "audio"},
            ),
        )

        response_id = interaction.id
        if not response_id or interaction.status != "completed":
            error_message = (
                f"[GEMINI] TTS interaction did not complete (status="
                f"{interaction.status}, errors={interaction.errors}). Retry the "
                f"request; if the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_tts] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        if interaction.output_audio is None:
            error_message = (
                "[GEMINI] Failed to extract audio bytes from TTS response: "
                "Gemini was unable to generate audio from the provided input. "
                "Ensure the input text is properly formatted and does not "
                "contain escape characters or unsupported control sequences. "
                "If the issue persists after input normalization, contact Kaapi."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_tts] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        if not interaction.output_audio.data:
            error_message = (
                "[GEMINI] TTS response is missing generated audio data. This is "
                "typically a Gemini server-side error. Wait a minute and retry; "
                "if the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_tts] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        raw_audio_bytes = base64.b64decode(interaction.output_audio.data)
        # Preview TTS models return headerless 24 kHz PCM (and reject audio/l16),
        # while 3.8+ return a full WAV; normalise to PCM for pcm_to_wav / convert_pcm_to_*.
        if raw_audio_bytes.startswith(b"RIFF"):
            with wave.open(io.BytesIO(raw_audio_bytes), "rb") as wav_file:
                raw_audio_bytes = wav_file.readframes(wav_file.getnframes())

        actual_format = "wav"
        if response_format == "mp3":
            converted_bytes, convert_error = convert_pcm_to_mp3(raw_audio_bytes)
            if convert_error:
                error_message = (
                    f"[KAAPI] Post-processing failure: unable to convert "
                    f"Gemini PCM audio to MP3 ({convert_error}). Falling "
                    f"back to WAV is possible by setting response_format='wav'."
                )
                logger.error(
                    f"[GoogleAIProvider._execute_tts] {error_message} | "
                    f"provider={provider}, model={model}, pcm_bytes={len(raw_audio_bytes)}"
                )
                return None, error_message
            encoded_content = base64.b64encode(converted_bytes or b"").decode("ascii")
            actual_format = "mp3"

        elif response_format == "ogg":
            converted_bytes, convert_error = convert_pcm_to_ogg(raw_audio_bytes)
            if convert_error:
                error_message = (
                    f"[KAAPI] Post-processing failure: unable to convert "
                    f"Gemini PCM audio to OGG ({convert_error}). Falling "
                    f"back to WAV is possible by setting response_format='wav'."
                )
                logger.error(
                    f"[GoogleAIProvider._execute_tts] {error_message} | "
                    f"provider={provider}, model={model}, pcm_bytes={len(raw_audio_bytes)}"
                )
                return None, error_message
            encoded_content = base64.b64encode(converted_bytes or b"").decode("ascii")
            actual_format = "ogg"

        else:
            if response_format and response_format != "wav":
                logger.warning(
                    f"[GoogleAIProvider._execute_tts] Unsupported response_format "
                    f"'{response_format}', returning native WAV | provider={provider}"
                )
            encoded_content = base64.b64encode(pcm_to_wav(raw_audio_bytes)).decode(
                "ascii"
            )

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response_id,
                model=interaction.model or model,
                provider=provider,
                output=AudioOutput(
                    content=AudioContent(
                        format="base64",
                        value=encoded_content,
                        mime_type=f"audio/{actual_format}",
                    )
                ),
            ),
            usage=self._extract_usage(interaction, provider),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = interaction.model_dump(mode="json")

        logger.info(
            f"[GoogleAIProvider._execute_tts] Successfully generated TTS response | "
            f"request_id={response_id}, provider={provider}, model={model}, audio_size={len(raw_audio_bytes)} bytes"
        )

        return llm_response, None

    def _execute_text(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str | list[ContentPart] | MultiModalInput,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        provider = completion_config.provider
        params = completion_config.params
        model = params.get("model") or DEFAULT_TEXT_MODELS["google"]

        if isinstance(resolved_input, MultiModalInput):
            content = self.format_parts(resolved_input.parts)
        elif isinstance(resolved_input, list):
            content = self.format_parts(resolved_input)
        else:
            content = [{"type": "text", "text": resolved_input}]

        instructions = params.get("instructions")
        # Kaapi configs carry thinking_config.thinking_level; native configs use reasoning.
        thinking_level = (params.get("thinking_config") or {}).get(
            "thinking_level"
        ) or params.get("reasoning")
        max_output_tokens = params.get("max_output_tokens")
        knowledge_base_ids = params.get("knowledge_base_ids")

        request: dict[str, Any] = {
            "model": model,
            "input": [{"type": "user_input", "content": content}],
        }
        if instructions:
            request["system_instruction"] = instructions

        generation_config: dict[str, Any] = {}
        if thinking_level:
            generation_config["thinking_level"] = thinking_level
        if max_output_tokens is not None:
            generation_config["max_output_tokens"] = max_output_tokens
        if generation_config:
            request["generation_config"] = generation_config

        if knowledge_base_ids:
            request["tools"] = [
                {"type": "file_search", "file_search_store_names": knowledge_base_ids}
            ]

        # create() is typed Interaction | Stream; we never pass stream=True.
        interaction = cast(Interaction, self.client.interactions.create(**request))

        response_id = interaction.id
        if not response_id or interaction.status != "completed":
            error_message = (
                f"[GEMINI] Text interaction did not complete (status="
                f"{interaction.status}, errors={interaction.errors}). This "
                f"typically means the response was blocked by Gemini's safety "
                f"filters or failed upstream. Review the prompt, then retry."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_text] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        if not interaction.output_text:
            error_message = (
                f"[GEMINI] Text response is missing generated content "
                f"(status={interaction.status}). This typically means the "
                f"model returned no text output — e.g. the response was blocked "
                f"by Gemini's safety filters or consumed by thinking tokens. "
                f"Review the prompt and token limits, then retry."
            )
            logger.warning(
                f"[GoogleAIProvider._execute_text] {error_message} | "
                f"provider={provider}, model={model}, response_id={response_id}"
            )
            return None, error_message

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response_id,
                model=interaction.model or model,
                provider=provider,
                output=TextOutput(content=TextContent(value=interaction.output_text)),
            ),
            usage=self._extract_usage(interaction, provider),
        )
        if include_provider_raw_response:
            llm_response.provider_raw_response = interaction.model_dump(mode="json")

        logger.info(
            f"[GoogleAIProvider._execute_text] Successfully generated text response | "
            f"request_id={response_id}, provider={provider}, model={model}"
        )
        return llm_response, None

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str | list[ContentPart] | MultiModalInput,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        provider = completion_config.provider
        completion_type = completion_config.type
        try:
            if completion_type == CompletionType.STT:
                return self._execute_stt(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )
            elif completion_type == CompletionType.TTS:
                return self._execute_tts(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )

            elif completion_type == CompletionType.TEXT:
                return self._execute_text(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )

        except TypeError as e:
            # handle unexpected arguments gracefully
            error_message = (
                f"[KAAPI] Invalid or unexpected parameter in Config: {str(e)}. "
                f"Review the completion config; one of the parameters does not "
                f"match the provider's expected signature."
            )
            logger.warning(
                f"[GoogleAIProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except interactions_errors.APIStatusError as e:
            code = e.status_code
            if code == 429:
                error_message = (
                    f"[GEMINI] Rate limit / quota exceeded (code: 429): "
                    f"{e.message}. You have hit Gemini's per-minute or per-day "
                    f"quota for this model. Wait at least 1 minute and retry; "
                    f"if the issue persists, request a quota increase from "
                    f"Google or contact Kaapi."
                )
            elif code in (401, 403):
                error_message = (
                    f"[GEMINI] Authentication / permission denied (code: "
                    f"{code}): {e.message}. Verify the Gemini API key is valid, "
                    f"not expired, and has access to the requested model and "
                    f"project."
                )
            elif code == 404:
                error_message = (
                    f"[GEMINI] Resource not found (code: 404): {e.message}. "
                    f"Check that the model name and any referenced IDs in "
                    f"your config are correct and available in your region."
                )
            elif code == 400:
                error_message = (
                    f"[GEMINI] Bad request (code: 400): {e.message}. Review "
                    f"your config parameters and input payload — the request "
                    f"shape, model, or content may be invalid for this Gemini "
                    f"endpoint."
                )
            elif code is not None and code >= 500:
                error_message = (
                    f"[GEMINI] Server error (code: {code}): {e.message}. This "
                    f"is typically transient (Gemini overloaded, internal "
                    f"error, or deadline exceeded) — retry in a few seconds. "
                    f"If the issue persists, contact Kaapi."
                )
            else:
                error_message = (
                    f"[GEMINI] Client error (code: {code}): {e.message}. "
                    f"Review the request configuration; if the issue persists, "
                    f"contact Kaapi."
                )
            logger.warning(
                f"[GoogleAIProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except interactions_errors.APIConnectionError as e:
            error_message = (
                f"[KAAPI] Could not reach Gemini ({type(e).__name__}): {e}. "
                f"Retry the request; if the issue persists, contact Kaapi."
            )
            logger.error(
                f"[GoogleAIProvider.execute] {error_message} | "
                f"provider={provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except genai_errors.ClientError as e:
            code = e.code
            status = e.status or ""
            msg = e.message or str(e)
            if code == 429:
                error_message = (
                    f"[GEMINI] Rate limit / quota exceeded (code: 429 "
                    f"{status}): {msg}. You have hit Gemini's per-minute or "
                    f"per-day quota for this model. Wait at least 1 minute "
                    f"and retry; if the issue persists, request a quota "
                    f"increase from Google or contact Kaapi."
                )
            elif code == 403:
                error_message = (
                    f"[GEMINI] Authentication / permission denied (code: 403 "
                    f"{status}): {msg}. Verify the Gemini API key is valid, "
                    f"not expired, and has access to the requested model and "
                    f"project."
                )
            elif code == 404:
                error_message = (
                    f"[GEMINI] Resource not found (code: 404 {status}): {msg}. "
                    f"Check that the model name and any referenced IDs in "
                    f"your config are correct and available in your region."
                )
            elif code == 400:
                error_message = (
                    f"[GEMINI] Bad request (code: 400 {status}): {msg}. "
                    f"Review your config parameters and input payload — the "
                    f"request shape, model, or content may be invalid for "
                    f"this Gemini endpoint."
                )
            else:
                error_message = (
                    f"[GEMINI] Client error (code: {code} {status}): {msg}. "
                    f"Review the request configuration; if the issue persists, "
                    f"contact Kaapi."
                )
            logger.warning(
                f"[GoogleAIProvider.execute] {error_message} | "
                f"provider={completion_config.provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except genai_errors.ServerError as e:
            error_message = (
                f"[GEMINI] Server error (code: {e.code} {e.status or ''}): "
                f"{e.message or str(e)}. This is typically transient (Gemini "
                f"overloaded, internal error, or deadline exceeded) — retry "
                f"in a few seconds. If the issue persists, contact Kaapi."
            )
            logger.error(
                f"[GoogleAIProvider.execute] {error_message} | "
                f"provider={completion_config.provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except genai_errors.UnknownApiResponseError as e:
            error_message = (
                f"[GEMINI] Returned a malformed or unparseable response: {e}. "
                f"This indicates an unexpected payload shape from Gemini — "
                f"retry the request. If the issue persists, contact Kaapi."
            )
            logger.warning(
                f"[GoogleAIProvider.execute] {error_message} | "
                f"provider={completion_config.provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except genai_errors.APIError as e:
            # Catch-all for any APIError subclass not handled above
            error_message = (
                f"[GEMINI] API error (code: {getattr(e, 'code', 'unknown')} "
                f"{getattr(e, 'status', '') or ''}): "
                f"{getattr(e, 'message', None) or str(e)}. If this persists, "
                f"contact Kaapi."
            )
            logger.warning(
                f"[GoogleAIProvider.execute] {error_message} | "
                f"provider={completion_config.provider}, type={completion_type}",
                exc_info=True,
            )
            return None, error_message

        except Exception as e:
            error_message = (
                f"[KAAPI] Unexpected error while executing Gemini "
                f"{completion_type or 'request'}: {str(e)}. This was not "
                f"raised by the Gemini SDK directly — likely a Kaapi-side "
                f"failure. Contact Kaapi if the issue persists."
            )
            logger.error(
                f"[GoogleAIProvider.execute] {error_message} | provider={provider}",
                exc_info=True,
            )
            return None, error_message

        error_message = (
            f"[KAAPI] Unsupported completion type '{completion_type}' for "
            f"Gemini. Use one of: text, stt, tts."
        )
        logger.warning(
            f"[GoogleAIProvider.execute] {error_message} | provider={provider}"
        )
        return None, error_message
