import logging
import base64
from typing import Any

from google import genai
from google.genai.types import (
    GenerateContentResponse,
    GenerateContentConfig,
    ThinkingConfig,
    SpeechConfig,
    VoiceConfig,
    PrebuiltVoiceConfig,
)

from app.models.llm import (
    NativeCompletionConfig,
    LLMCallResponse,
    QueryParams,
    LLMResponse,
    Usage,
    TextOutput,
    TextContent,
    ImageContent,
    PDFContent,
)
from app.models.llm.constants import (
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
)
from app.models.llm.response import AudioOutput, AudioContent
from app.services.llm.providers.base import BaseProvider, ContentPart, MultiModalInput
from app.services.llm.mappers import BCP47_LOCALE_TO_GEMINI_LANG
from app.core.audio_utils import (
    AudioRef,
    convert_pcm_to_mp3,
    convert_pcm_to_ogg,
    pcm_to_wav,
)

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
    ) -> list[dict]:
        items = []
        for part in parts:
            if isinstance(part, TextContent):
                items.append({"text": part.value})

            elif isinstance(part, ImageContent):
                if part.format == "base64":
                    items.append(
                        {
                            "inline_data": {
                                "data": part.value,
                                "mime_type": part.mime_type,
                            }
                        }
                    )
                else:
                    items.append(
                        {
                            "file_data": {
                                "file_uri": part.value,
                                "mime_type": part.mime_type,
                                "display_name": None,
                            }
                        }
                    )
            elif isinstance(part, PDFContent):
                if part.format == "base64":
                    items.append(
                        {
                            "inline_data": {
                                "data": part.value,
                                "mime_type": part.mime_type,
                            }
                        }
                    )
                else:
                    items.append(
                        {
                            "file_data": {
                                "file_uri": part.value,
                                "mime_type": part.mime_type,
                                "display_name": None,
                            }
                        }
                    )
        return items

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
            return None, f"{provider} STT requires AudioRef input"

        model = generation_params.get("model") or DEFAULT_STT_MODEL
        instructions = generation_params.get("instructions", "")
        input_language = generation_params.get("input_language") or "auto"
        output_language = generation_params.get("output_language", "")
        temperature = generation_params.get("temperature") or 0.0

        # Build transcription/translation instruction
        if input_language == "auto":
            lang_instruction = (
                "Detect the spoken language automatically and transcribe the audio"
            )
        else:
            lang_instruction = f"Transcribe the audio from {input_language} in the native script of {input_language}"

        if output_language and output_language != input_language:
            lang_instruction += f" and translate to {output_language} in the native script of {output_language} and only return transcribed script in {output_language}."

        forced_transcription_text = "Only return transcribed text and no other text."
        # Merge user instructions with language instructions
        if instructions:
            merged_instruction = (
                f"{instructions}. {lang_instruction}. {forced_transcription_text}"
            )
        else:
            merged_instruction = f"{lang_instruction}. {forced_transcription_text}"

        logger.info(
            f"The merged instructions is {merged_instruction} and output language is {output_language} and input language is {input_language}"
        )

        # Materialize the AudioRef to a temp file so the genai SDK can upload it.
        with resolved_input.to_path() as audio_path:
            gemini_file = self.client.files.upload(file=audio_path)

        contents = []
        if merged_instruction:
            contents.append(merged_instruction)
        contents.append(gemini_file)

        response: GenerateContentResponse = self.client.models.generate_content(
            model=model,
            contents=contents,
            # switch back default thinking configs for reasoning supported models in future
            config=GenerateContentConfig(
                thinking_config=ThinkingConfig(
                    include_thoughts=True, thinking_budget=1000
                ),
                temperature=temperature,
            ),
        )

        # Validate response has required fields
        if not response.response_id:
            return None, "Google AI response missing response_id"

        if not response.text:
            return None, "Google AI response missing text content"

        # Extract usage metadata with null checks
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            total_tokens = response.usage_metadata.total_token_count or 0
            reasoning_tokens = response.usage_metadata.thoughts_token_count or 0
        else:
            logger.warning(
                f"[GoogleAIProvider._execute_stt] Response missing usage_metadata, using zeros | provider={provider}"
            )
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            reasoning_tokens = 0

        # Build response
        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response.response_id,
                model=response.model_version or model,
                provider=provider,
                output=TextOutput(content=TextContent(value=response.text)),
            ),
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = response.model_dump()

        logger.info(
            f"[GoogleAIProvider._execute_stt] Successfully generated STT response | "
            f"request_id={response.response_id}, provider={provider}, model={model}"
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

        # Validate input is a text string
        if not isinstance(resolved_input, str):
            return None, f"{provider} TTS requires text string as input"

        if not resolved_input.strip():
            return None, "Text input cannot be empty"

        # Extract params with defaults (language is optional — Gemini auto-detects from script)
        model = generation_params.get("model") or DEFAULT_TTS_MODEL
        voice = generation_params.get("voice") or DEFAULT_TTS_VOICE

        language = generation_params.get("language")

        # Extract optional params
        response_format = generation_params.get("response_format", "wav")

        # Extract Gemini-specific params from provider_specific.gemini
        provider_specific = generation_params.get("provider_specific", {})
        gemini_params = provider_specific.get("gemini", {})

        director_notes = gemini_params.get("director_notes", "")
        # Build Gemini TTS config
        config_kwargs = {
            "response_modalities": ["AUDIO"],
            "speech_config": SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice)
                ),
                language_code=language,
            ),
        }

        if director_notes:
            config_kwargs["system_instruction"] = director_notes

        config = GenerateContentConfig(**config_kwargs)

        # Execute TTS
        response: GenerateContentResponse = self.client.models.generate_content(
            model=model, contents=resolved_input, config=config
        )
        if not response.response_id:
            return None, "Google AI response missing response_id"
        try:
            raw_audio_bytes = response.candidates[0].content.parts[0].inline_data.data

        except (IndexError, AttributeError) as e:
            return None, f"Failed to extract audio from response: {str(e)}"

        if not raw_audio_bytes:
            return None, "Google AI response missing audio data"

        # Post-process audio format conversion if needed
        # Gemini TTS natively outputs 24kHz 16-bit raw PCM — wrap in WAV container
        actual_format = "wav"
        wav_bytes = pcm_to_wav(raw_audio_bytes)
        encoded_content = base64.b64encode(wav_bytes).decode("ascii")

        if response_format and response_format != "wav":
            # Need to convert from WAV to requested format
            logger.info(
                f"[GoogleAIProvider._execute_tts] Converting audio from WAV to {response_format} | provider={provider}"
            )

            if response_format == "mp3":
                converted_bytes, convert_error = convert_pcm_to_mp3(raw_audio_bytes)
                if convert_error:
                    return None, f"Failed to convert audio to MP3: {convert_error}"
                encoded_content = base64.b64encode(converted_bytes or b"").decode(
                    "ascii"
                )
                actual_format = "mp3"

            elif response_format == "ogg":
                converted_bytes, convert_error = convert_pcm_to_ogg(raw_audio_bytes)
                if convert_error:
                    return None, f"Failed to convert audio to OGG: {convert_error}"
                encoded_content = base64.b64encode(converted_bytes or b"").decode(
                    "ascii"
                )
                actual_format = "ogg"
            else:
                logger.warning(
                    f"[GoogleAIProvider._execute_tts] Unsupported response_format '{response_format}', returning native WAV | provider={provider}"
                )
                response_format = "wav"
            logger.info(
                f"[GoogleAIProvider._execute_tts] Audio conversion successful: {actual_format.upper()} ({len(raw_audio_bytes)} bytes) | provider={provider}"
            )
        response_mime_type = f"audio/{response_format}"

        # Extract usage metadata
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            total_tokens = response.usage_metadata.total_token_count or 0
            reasoning_tokens = response.usage_metadata.thoughts_token_count or 0
        else:
            logger.warning(
                f"[GoogleAIProvider._execute_tts] Response missing usage_metadata, using zeros | provider={provider}"
            )
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            reasoning_tokens = 0

        # Build response
        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response.response_id,
                model=response.model_version or model,
                provider=provider,
                # output=LLMOutput(audio_bytes=audio_bytes, audio_format=actual_format),
                output=AudioOutput(
                    content=AudioContent(
                        format="base64",
                        value=encoded_content,
                        mime_type=response_mime_type,
                    )
                ),
            ),
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
        )

        if include_provider_raw_response:
            llm_response.provider_raw_response = response.model_dump()

        logger.info(
            f"[GoogleAIProvider._execute_tts] Successfully generated TTS response | "
            f"request_id={response.response_id}, provider={provider}, model={model}, audio_size={len(raw_audio_bytes)} bytes"
        )

        return llm_response, None

    def _execute_text(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str | list[ContentPart] | MultiModalInput,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        model = completion_config.params.get("model")
        if not model:
            return None, "Missing 'model' in native params"

        if isinstance(resolved_input, MultiModalInput):
            gemini_parts = self.format_parts(resolved_input.parts)
            contents = [{"role": "user", "parts": gemini_parts}]
        elif isinstance(resolved_input, list):
            gemini_parts = self.format_parts(resolved_input)
            contents = [{"role": "user", "parts": gemini_parts}]
        else:
            contents = [{"role": "user", "parts": [{"text": resolved_input}]}]

        instructions = completion_config.params.get("instructions", "")
        temperature = completion_config.params.get("temperature", None)
        thinking_level = completion_config.params.get("reasoning", None)

        generation_kwargs = {}
        if instructions:
            generation_kwargs["system_instruction"] = instructions

        if temperature is not None:
            generation_kwargs["temperature"] = temperature

        if thinking_level is not None:
            generation_kwargs["thinking_config"] = ThinkingConfig(
                include_thoughts=False, thinking_level=thinking_level
            )

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=GenerateContentConfig(**generation_kwargs),
        )

        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            total_tokens = response.usage_metadata.total_token_count or 0
            reasoning_tokens = response.usage_metadata.thoughts_token_count or 0
        else:
            logger.warning(
                f"[GoogleAIProvider._execute_text] Response missing usage_metadata, using zeros | provider={completion_config.provider}"
            )
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            reasoning_tokens = 0

        llm_response = LLMCallResponse(
            response=LLMResponse(
                provider_response_id=response.response_id,
                model=response.model_version or model,
                provider=completion_config.provider,
                output=TextOutput(content=TextContent(value=response.text)),
            ),
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
        )
        if include_provider_raw_response:
            llm_response.provider_raw_response = response.model_dump(mode="json")

        logger.info(
            f"[GoogleAIProvider._execute_text] Successfully generated text response | "
            f"request_id={response.response_id}, provider={completion_config.provider}, model={model}"
        )
        return llm_response, None

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str | list[ContentPart] | MultiModalInput,
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

            elif completion_type == "text":
                return self._execute_text(
                    completion_config=completion_config,
                    resolved_input=resolved_input,
                    include_provider_raw_response=include_provider_raw_response,
                )

        except TypeError as e:
            # handle unexpected arguments gracefully
            error_message = f"Invalid or unexpected parameter in Config: {str(e)}"
            return None, error_message

        except Exception as e:
            error_message = "Unexpected error occurred"
            logger.error(
                f"[GoogleAIProvider.execute] {error_message}: {str(e)} | provider={completion_config.provider}",
                exc_info=True,
            )
            return None, error_message
