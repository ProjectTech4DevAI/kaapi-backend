import logging

from sqlmodel import Session

from app.crud.model_config import is_reasoning_model
from app.models.llm import KaapiCompletionConfig, NativeCompletionConfig
from app.models.llm.constants import (
    BCP47_LOCALE_TO_GEMINI_LANG,
    BCP47_TO_ELEVENLABS_LANG,
    DEFAULT_ELEVENLABS_STT_MODEL,
    DEFAULT_ELEVENLABS_TTS_MODEL,
    DEFAULT_SARVAM_STT_MODEL,
    DEFAULT_SARVAM_TTS_MODEL,
    DEFAULT_TEXT_MODELS,
    DEFAULT_TTS_VOICE,
    ELEVENLABS_VOICE_TO_ID,
    CompletionType,
    Provider,
)

SARVAM_DEFAULTS_BY_TYPE = {
    "stt": DEFAULT_SARVAM_STT_MODEL,
    "tts": DEFAULT_SARVAM_TTS_MODEL,
}
ELEVENLABS_DEFAULTS_BY_TYPE = {
    "stt": DEFAULT_ELEVENLABS_STT_MODEL,
    "tts": DEFAULT_ELEVENLABS_TTS_MODEL,
}

logger = logging.getLogger(__name__)


def voice_to_id(voice: str) -> str | None:
    """
        Convert voice to its corresponding voice_id

    Returns:
        voice_id associated with a voice
    """

    return ELEVENLABS_VOICE_TO_ID.get(voice)


def bcp47_to_elevenlabs_lang(bcp47_code: str) -> str | None:
    """Convert a BCP-47 language tag to an ElevenLabs ISO 639-3 language code.

    Args:
        bcp47_code: BCP-47 language tag (e.g. "en-IN", "hi-IN", "ta-IN")

    Returns:
        ISO 639-1 code (e.g. "en", "hi", "ta") or None if unsupported
    """
    return BCP47_TO_ELEVENLABS_LANG.get(bcp47_code)


def map_kaapi_to_openai_params(
    session: Session, kaapi_params: dict
) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to OpenAI API parameters.

    This mapper transforms standardized Kaapi parameters into OpenAI-specific
    parameter format, enabling provider-agnostic interface design.

    Args:
        session: Database session used to look up the model's config
        kaapi_params: Dictionary with standardized Kaapi parameters

    Supported Mapping:
        - model → model
        - instructions → instructions
        - knowledge_base_ids → tools[file_search].vector_store_ids
        - max_num_results → tools[file_search].max_num_results (fallback default)
        - effort → reasoning.effort (if reasoning supported by model else suppressed)
        - reasoning → legacy alias for effort, used only when effort is absent
        - temperature → temperature (if reasoning not supported by model else suppressed)

    Returns:
        Tuple of:
        - Dictionary of OpenAI API parameters ready to be passed to the API
        - List of warnings describing suppressed or ignored parameters
    """
    openai_params = {}
    warnings = []

    model = kaapi_params.get("model")
    reasoning = kaapi_params.get("reasoning")
    effort = kaapi_params.get("effort")
    temperature = kaapi_params.get("temperature")
    instructions = kaapi_params.get("instructions")
    knowledge_base_ids = kaapi_params.get("knowledge_base_ids")
    max_num_results = kaapi_params.get("max_num_results")

    support_reasoning = bool(model) and is_reasoning_model(
        session=session, provider="openai", model_name=model
    )

    # 'effort' is the canonical knob; 'reasoning' is a legacy alias kept for
    # backward compatibility and only consulted when 'effort' is absent
    effort_value = effort if effort is not None else reasoning

    # Handle reasoning vs temperature mutual exclusivity
    if support_reasoning:
        if effort_value is not None:
            openai_params["reasoning"] = {"effort": effort_value}

        if temperature is not None:
            warnings.append(
                "Parameter 'temperature' was suppressed because the selected model "
                "supports reasoning, and temperature is ignored when reasoning is enabled."
            )
    else:
        if effort_value is not None:
            warnings.append(
                "Parameters 'reasoning'/'effort' were suppressed because the "
                "selected model does not support reasoning."
            )

        if temperature is not None:
            openai_params["temperature"] = temperature

    openai_params["model"] = model or DEFAULT_TEXT_MODELS["openai"]

    if instructions:
        openai_params["instructions"] = instructions

    if knowledge_base_ids:
        openai_params["tools"] = [
            {
                "type": "file_search",
                "vector_store_ids": knowledge_base_ids,
                "max_num_results": max_num_results or 20,
            }
        ]

    return openai_params, warnings


def map_kaapi_to_google_params(
    kaapi_params: dict, completion_type: str
) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to Google AI (Gemini) API parameters.

    This mapper transforms standardized Kaapi parameters into Google-specific
    parameter format for the Gemini API.

    Args:
        kaapi_params: Dictionary with standardized Kaapi parameters
        completion_type: Type of completion ("text", "stt", or "tts")

    Supported Mapping:
        - model → model
        - instructions → instructions (for STT prompts, if available)
        - temperature -> temperature parameter (0-2)

    Returns:
        Tuple of:
        - Dictionary of Google AI API parameters ready to be passed to the API
        - List of warnings describing suppressed or ignored parameters
    """
    google_params = {}
    warnings = []

    # Model is present in all param types; text falls back to the centralized
    # default. STT/TTS require an explicit model (Gemini variant differs by mode).
    model = kaapi_params.get("model")
    if not model and completion_type == CompletionType.TEXT:
        model = DEFAULT_TEXT_MODELS["google"]
    if not model:
        return {}, ["Missing required 'model' parameter"]

    google_params["model"] = model

    if completion_type == CompletionType.TEXT:
        # Text completion - instructions, temperature, reasoning, knowledge_base_ids
        instructions = kaapi_params.get("instructions")
        if instructions:
            google_params["instructions"] = instructions

        temperature = kaapi_params.get("temperature")
        if temperature is not None:
            google_params["temperature"] = temperature

        reasoning = kaapi_params.get("reasoning")
        if reasoning:
            google_params["reasoning"] = reasoning

        # Warn about unsupported parameters
        if kaapi_params.get("knowledge_base_ids"):
            # TODO: Will take up later, when we add google filesearch tool support
            warnings.append(
                "Parameter 'knowledge_base_ids' is not supported by Google AI and was ignored."
            )

    elif completion_type == CompletionType.TTS:
        # TTS mode - voice, language, response_format
        # Apply smart defaults for voice and response_format (following ElevenLabs pattern)
        voice = kaapi_params.get("voice") or DEFAULT_TTS_VOICE
        google_params["voice"] = voice

        response_format = kaapi_params.get("response_format") or "wav"
        google_params["response_format"] = response_format

        # Language: Only set if explicitly provided (None/missing = auto-detect)
        language = kaapi_params.get("language")
        if language:
            google_lang = BCP47_LOCALE_TO_GEMINI_LANG.get(language) or None
            if not google_lang:
                warnings.append(
                    f"Unsupported language '{language}' for Gemini TTS, using auto-detect"
                )
            else:
                google_params["language"] = google_lang

    elif completion_type == CompletionType.STT:
        # STT mode - instructions, temperature, input_language, output_language, response_format
        # Apply smart default for input_language
        input_language = kaapi_params.get("input_language") or "auto"
        google_params["input_language"] = input_language

        # Optional parameters - only set if provided
        instructions = kaapi_params.get("instructions")
        if instructions:
            google_params["instructions"] = instructions

        temperature = kaapi_params.get("temperature")
        if temperature is not None:
            google_params["temperature"] = temperature

        output_language = kaapi_params.get("output_language")
        if output_language:
            google_params["output_language"] = output_language

        response_format = kaapi_params.get("response_format")
        if response_format:
            google_params["response_format"] = response_format

    else:
        return {}, [f"Unsupported completion type '{completion_type}' for Google AI"]

    return google_params, warnings


def map_kaapi_to_sarvam_params(
    kaapi_params: dict, completion_type: str
) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to SarvamAI API parameters.

    Handles both STTLLMParams and TTSLLMParams.

    STTLLMParams: model, instructions, input_language, output_language, response_format, temperature
    TTSLLMParams: model, voice, language, response_format

    Args:
        kaapi_params: Dictionary with standardized Kaapi parameters
        completion_type: Type of completion ("stt" or "tts")

    Returns:
        Tuple of:
        - Dictionary of SarvamAI API parameters
        - List of warnings for unsupported parameters
    """
    sarvam_params = {}
    warnings = []

    # Model falls back to the per-type Sarvam default.
    model = kaapi_params.get("model") or SARVAM_DEFAULTS_BY_TYPE.get(completion_type)
    if not model:
        return {}, [f"Unsupported completion type '{completion_type}' for SarvamAI"]
    sarvam_params["model"] = model

    if completion_type == CompletionType.TTS:
        # TTS mode - map TTSLLMParams
        # Required: target_language_code (API requirement)
        language = kaapi_params.get("language")
        if not language:
            return {}, ["Missing required 'language' parameter for TTS"]
        sarvam_params["target_language_code"] = language

        # Optional: speaker (has API default: Shubh for v3, Anushka for v2)
        voice = kaapi_params.get("voice")
        if voice:
            sarvam_params["speaker"] = voice

        # Optional: output_audio_codec
        response_format = kaapi_params.get("response_format")
        if response_format:
            # Map audio format to SarvamAI codec
            # Supported: mp3, linear16, mulaw, alaw, opus, flac, aac, wav
            format_mapping = {
                "mp3": "mp3",
                "wav": "wav",
                "ogg": "opus",  # Map ogg to opus (closest match)
            }
            sarvam_params["output_audio_codec"] = format_mapping.get(
                response_format, "wav"
            )

    elif completion_type == CompletionType.STT:
        # STT mode - map STTLLMParams
        input_language = kaapi_params.get("input_language")
        output_language = kaapi_params.get("output_language")

        # Set language_code (optional, defaults to "unknown" for auto-detection)
        if input_language == "auto":
            sarvam_params["language_code"] = "unknown"
        elif input_language:
            sarvam_params["language_code"] = input_language
        else:
            # Default to "unknown" for auto-detection if not provided
            sarvam_params["language_code"] = "unknown"

        # Set mode only for saaras:v3 model (not for saarika:v2.5)
        # mode parameter: transcribe, translate, verbatim, translit, or codemix
        if model and "saaras" in model:
            transcription_mode = "transcribe"

            if output_language is None:
                output_language = input_language

            if output_language == "en-IN" and input_language != output_language:
                transcription_mode = "translate"

            sarvam_params["mode"] = transcription_mode

        # Warn about unsupported STT parameters
        instructions = kaapi_params.get("instructions")
        if instructions:
            warnings.append(
                "Parameter 'instructions' is not supported by SarvamAI STT and was ignored"
            )

        temperature = kaapi_params.get("temperature")
        if temperature is not None:
            warnings.append(
                "Parameter 'temperature' is not supported by SarvamAI STT and was ignored"
            )

        response_format = kaapi_params.get("response_format")
        if response_format:
            warnings.append(
                "Parameter 'response_format' is not supported by SarvamAI STT and was ignored"
            )

    else:
        return {}, [f"Unsupported completion type '{completion_type}' for SarvamAI"]
    logger.info(f"Sarvam params {sarvam_params}")
    return sarvam_params, warnings


def map_kaapi_to_elevenlabs_params(
    kaapi_params: dict, completion_type: str
) -> tuple[dict, list[str]]:
    """
    Map Kaapi-abstracted parameters to ElevenLab API params
    Handles both STTLLMParams and TTSLLMParams.

    STTLLMParams: model, instructions, input_language, output_language, response_format, temperature
    TTSLLMParams: model, voice, language, response_format

    Args:
        kaapi_params: Dictionary with standardized Kaapi parameters
        completion_type: Type of completion ("stt" or "tts")

    Returns:
        Tuple of:
        - Dictionary of ELevenlabs API parameters
        - List of warnings for unsupported parameters

    """
    elevenlabs_params = {}
    warnings = []

    model_id = kaapi_params.get("model") or ELEVENLABS_DEFAULTS_BY_TYPE.get(
        completion_type
    )
    if not model_id:
        return {}, [f"Unsupported completion type '{completion_type}' for ElevenLabs"]
    elevenlabs_params["model_id"] = model_id

    if completion_type == CompletionType.TTS:
        # TTS Mode - map TTSLLMParams
        voice = kaapi_params.get("voice")
        if not voice:
            return {}, ["Missing required 'voice' parameter for TTS"]

        voice_id = voice_to_id(voice)
        if not voice_id:
            return {}, [f"Unsupported voice '{voice}' for ElevenLabs TTS"]
        elevenlabs_params["voice_id"] = voice_id

        language = kaapi_params.get("language")
        if language:
            elevenlabs_lang = bcp47_to_elevenlabs_lang(language)
            if not elevenlabs_lang:
                warnings.append(
                    f"Unsupported language '{language}' for ElevenLabs TTS, using default"
                )
            else:
                elevenlabs_params["language_code"] = elevenlabs_lang

        response_format = kaapi_params.get("response_format")
        if response_format:
            # Map audio format to Elevenlabs codec
            # supports mp3, wav and opus (ogg maps to opus)
            format_mapping = {
                "mp3": "mp3_44100_128",
                "wav": "wav_24000",
                "ogg": "opus_48000_128",  # Map ogg to opus
            }
            elevenlabs_params["output_format"] = format_mapping.get(
                response_format, "wav_24000"
            )

    elif completion_type == CompletionType.STT:
        # STT mode - map STTLLMParams
        input_language = kaapi_params.get("input_language")
        output_language = kaapi_params.get("output_language")

        if input_language == "auto":
            elevenlabs_params["language_code"] = None
        elif input_language:
            elevenlabs_lang = bcp47_to_elevenlabs_lang(input_language)
            if elevenlabs_lang:
                elevenlabs_params["language_code"] = elevenlabs_lang
            else:
                warnings.append(
                    f"Unsupported language '{input_language}' for ElevenLabs STT, defaulting to auto-detect"
                )

        if output_language and output_language != input_language:
            warnings.append(
                "Parameter 'output_language' is not supported by ElevenLabs STT. "
                "ElevenLabs only supports transcription, not translation. "
                "The audio will be transcribed in its original language."
            )

        temperature = kaapi_params.get("temperature")
        if temperature is not None:
            elevenlabs_params["temperature"] = temperature

        response_format = kaapi_params.get("response_format")
        if response_format:
            warnings.append("Kaapi only supports 'txt' as the default response format.")

        # Warn about unsupported STT parameters
        instructions = kaapi_params.get("instructions")
        if instructions:
            warnings.append(
                "Parameter 'instructions' is not supported by ElevenLabs STT and was ignored."
            )
    else:
        return {}, [f"Unsupported completion type '{completion_type}' for ElevenLabs"]

    return elevenlabs_params, warnings


def map_kaapi_to_anthropic_params(
    kaapi_params: dict,
) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to Anthropic Messages API parameters.

    Supported Mapping:
        - model → model
        - instructions → system
        - temperature → temperature
        - top_p → top_p
        - max_output_tokens → max_tokens (Anthropic requires this;
          provider defaults if absent)

    Unsupported Kaapi params:
        - knowledge_base_ids / max_num_results: Anthropic has no native
          vector-store / file_search tool, dropped with warning.
        - reasoning / effort / summary: Messages API does not expose a
          reasoning-effort knob, dropped with warning.
    """
    anthropic_params: dict = {}
    warnings: list[str] = []

    model = kaapi_params.get("model")
    instructions = kaapi_params.get("instructions")
    temperature = kaapi_params.get("temperature")
    top_p = kaapi_params.get("top_p")
    max_output_tokens = kaapi_params.get("max_output_tokens")
    knowledge_base_ids = kaapi_params.get("knowledge_base_ids")
    reasoning = kaapi_params.get("reasoning")
    effort = kaapi_params.get("effort")
    summary = kaapi_params.get("summary")

    anthropic_params["model"] = model or DEFAULT_TEXT_MODELS["anthropic"]

    if instructions:
        anthropic_params["system"] = instructions

    if temperature is not None:
        anthropic_params["temperature"] = temperature

    if top_p is not None:
        anthropic_params["top_p"] = top_p

    if max_output_tokens is not None:
        anthropic_params["max_tokens"] = max_output_tokens

    if knowledge_base_ids:
        warnings.append(
            "Parameter 'knowledge_base_ids' was ignored because Anthropic has no "
            "native vector-store/file_search tool. Inline document content blocks instead."
        )

    if reasoning is not None or effort is not None or summary is not None:
        warnings.append(
            "Parameters 'reasoning'/'effort'/'summary' were ignored because the "
            "Anthropic Messages API does not expose a reasoning-effort knob."
        )

    return anthropic_params, warnings


def transform_kaapi_config_to_native(
    session: Session,
    kaapi_config: KaapiCompletionConfig,
) -> tuple[NativeCompletionConfig, list[str]]:
    """Transform Kaapi completion config to native provider config with mapped parameters.

    Supports OpenAI,Google AI and Sarvam AI providers.

    Args:
        session: Database session used to look up model-specific config (e.g. reasoning support)
        kaapi_config: KaapiCompletionConfig with abstracted parameters

    Returns:
        Tuple of:
        - NativeCompletionConfig with provider-native parameters ready for API
        - List of warnings for suppressed/ignored parameters
    """
    if kaapi_config.provider == Provider.OPENAI:
        mapped_params, warnings = map_kaapi_to_openai_params(
            session=session, kaapi_params=kaapi_config.params
        )
        return (
            NativeCompletionConfig(
                provider="openai-native", params=mapped_params, type=kaapi_config.type
            ),
            warnings,
        )

    if kaapi_config.provider == Provider.GOOGLE_AISTUDIO:
        mapped_params, warnings = map_kaapi_to_google_params(
            kaapi_config.params, kaapi_config.type
        )
        return (
            NativeCompletionConfig(
                provider="google-aistudio-native",
                params=mapped_params,
                type=kaapi_config.type,
            ),
            warnings,
        )

    if kaapi_config.provider == Provider.SARVAMAI:
        mapped_params, warnings = map_kaapi_to_sarvam_params(
            kaapi_config.params, kaapi_config.type
        )
        return (
            NativeCompletionConfig(
                provider="sarvamai-native", params=mapped_params, type=kaapi_config.type
            ),
            warnings,
        )

    if kaapi_config.provider == Provider.ELEVENLABS:
        mapped_params, warnings = map_kaapi_to_elevenlabs_params(
            kaapi_config.params, kaapi_config.type
        )
        return (
            NativeCompletionConfig(
                provider="elevenlabs-native",
                params=mapped_params,
                type=kaapi_config.type,
            ),
            warnings,
        )

    if kaapi_config.provider == Provider.GOOGLE:
        if kaapi_config.type not in (CompletionType.STT, CompletionType.TTS):
            raise ValueError(
                f"google provider does not support completion type '{kaapi_config.type}'. "
                "Use the 'google-aistudio' provider for text completions."
            )
        # Kaapi STT/TTS param shape is identical to Google's; reuse the Google mapper.
        mapped_params, warnings = map_kaapi_to_google_params(
            kaapi_config.params, kaapi_config.type
        )
        return (
            NativeCompletionConfig(
                provider="google-native",
                params=mapped_params,
                type=kaapi_config.type,
            ),
            warnings,
        )

    if kaapi_config.provider == Provider.ANTHROPIC:
        if kaapi_config.type != CompletionType.TEXT:
            raise ValueError(
                f"Anthropic provider does not support completion type '{kaapi_config.type}'"
            )
        mapped_params, warnings = map_kaapi_to_anthropic_params(kaapi_config.params)
        return (
            NativeCompletionConfig(
                provider="anthropic-native",
                params=mapped_params,
                type=kaapi_config.type,
            ),
            warnings,
        )

    raise ValueError(f"Unsupported provider: {kaapi_config.provider}")
