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
)
from app.models.llm.response import AudioOutput, AudioContent
from app.services.llm.providers.base import BaseProvider
from app.core.audio_utils import convert_pcm_to_mp3, convert_pcm_to_ogg

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

    def _execute_stt(
        self,
        completion_config: NativeCompletionConfig,
        resolved_input: str,
        include_provider_raw_response: bool = False,
    ) -> tuple[LLMCallResponse | None, str | None]:
        """Execute speech-to-text completion using Google AI.

        Args:
            completion_config: Configuration for the completion request
            resolved_input: File path to the audio input
            include_provider_raw_response: Whether to include raw provider response

        Returns:
            Tuple of (LLMCallResponse, error_message)
        """
        provider = completion_config.provider
        generation_params = completion_config.params
        # Validate input is a file path string
        if not isinstance(resolved_input, str):
            return None, f"{provider} STT requires file path as string"

        model = generation_params.get("model")
        if not model:
            return None, "Missing 'model' in native params"

        instructions = generation_params.get("instructions", "")
        input_language = generation_params.get("input_language") or "auto"
        output_language = generation_params.get("output_language", "")
        temperature = generation_params.get("temperature", 0.7)

        # Build transcription/translation instruction
        if input_language == "auto":
            lang_instruction = (
                "Detect the spoken language automatically and transcribe the audio"
            )
        else:
            lang_instruction = f"Transcribe the audio from {input_language} in the native script of {input_language}"

        if output_language and output_language != input_language:
            lang_instruction += f" and translate to {output_language} in the native script of {output_language}"

        forced_transcription_text = "Only return transcribed text and no other text."
        # Merge user instructions with language instructions
        if instructions:
            merged_instruction = (
                f"{instructions}. {lang_instruction}. {forced_transcription_text}"
            )
        else:
            merged_instruction = f"{lang_instruction}. {forced_transcription_text}"

        # Upload file and generate content
        gemini_file = self.client.files.upload(file=resolved_input)

        contents = []
        if merged_instruction:
            contents.append(merged_instruction)
        contents.append(gemini_file)

        response: GenerateContentResponse = self.client.models.generate_content(
            model=model,
            contents=contents,
            # switch back default thinking configs for reasoning supported models in future
            config=GenerateContentConfig(
                # thinking_config=ThinkingConfig(thinking_level="low"),
                temperature=temperature
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
                f"[GoogleAIProvider._execute_stt] Response missing usage_metadata, using zeros"
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
            f"[GoogleAIProvider._execute_stt] Successfully generated STT response: {response.response_id}"
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

        # Extract required params
        model = generation_params.get("model")
        if not model:
            return None, "Missing 'model' in native params"

        voice = generation_params.get("voice")
        if not voice:
            return None, "Missing 'voice' in native params"

        language = generation_params.get("language")
        if not language:
            return None, "Missing 'language' in native params"

        # Extract optional params
        response_format = generation_params.get("response_format", "wav")

        # Extract Gemini-specific params from provider_specific.gemini
        provider_specific = generation_params.get("provider_specific", {})
        gemini_params = provider_specific.get("gemini", {})

        director_notes = gemini_params.get("director_notes")
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
        # Gemini TTS natively outputs 24kHz 16-bit PCM (WAV format)
        actual_format = "wav"  # Native Gemini TTS output format
        encoded_content = base64.b64encode(raw_audio_bytes or b"").decode("ascii")

        if response_format and response_format != "wav":
            # Need to convert from WAV to requested format
            logger.info(
                f"[GoogleAIProvider._execute_tts] Converting audio from WAV to {response_format}"
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
                    f"[GoogleAIProvider._execute_tts] Unsupported response_format '{response_format}', returning native WAV"
                )
                response_format = "wav"
            logger.info(
                f"[GoogleAIProvider._execute_tts] Audio conversion successful: {actual_format.upper()} ({len(raw_audio_bytes)} bytes)"
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
                f"[GoogleAIProvider._execute_tts] Response missing usage_metadata, using zeros"
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
            f"[GoogleAIProvider._execute_tts] Successfully generated TTS response: "
            f"{response.response_id}, audio_size={len(raw_audio_bytes)} bytes"
        )

        return llm_response, None

    def execute(
        self,
        completion_config: NativeCompletionConfig,
        query: QueryParams,  # Not used by Google AI provider (no conversation support yet)
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
                    f"Unsupported completion type '{completion_type}' for Google AI provider",
                )

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
