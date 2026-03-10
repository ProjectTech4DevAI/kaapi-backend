"""Parameter mappers for converting Kaapi-abstracted parameters to provider-specific formats."""

import litellm
from app.models.llm import KaapiCompletionConfig, NativeCompletionConfig

# BCP-47 language tag → ElevenLabs ISO 639-3 code (Indic + English)
BCP47_TO_ELEVENLABS_LANG: dict[str, str] = {
    "en-IN": "eng",
    "hi-IN": "hin",
    "bn-IN": "ben",
    "ta-IN": "tam",
    "te-IN": "tel",
    "mr-IN": "mar",
    "gu-IN": "guj",
    "kn-IN": "kan",
    "ml-IN": "mal",
    "pa-IN": "pan",
    "od-IN": "ori",
    "as-IN": "asm",
    "ur-IN": "urd",
    "ne-IN": "nep",
    "sd-IN": "snd",
}

ELEVENLABS_VOICE_TO_ID: dict[str, str] = {
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Callum": "N2lVS1w4EtoT3dr4eOWO",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
}


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
        ISO 639-3 code (e.g. "eng", "hin", "tam") or None if unsupported
    """
    return BCP47_TO_ELEVENLABS_LANG.get(bcp47_code)


def map_kaapi_to_openai_params(kaapi_params: dict) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to OpenAI API parameters.

    This mapper transforms standardized Kaapi parameters into OpenAI-specific
    parameter format, enabling provider-agnostic interface design.

    Args:
        kaapi_params: Dictionary with standardized Kaapi parameters

    Supported Mapping:
        - model → model
        - instructions → instructions
        - knowledge_base_ids → tools[file_search].vector_store_ids
        - max_num_results → tools[file_search].max_num_results (fallback default)
        - reasoning → reasoning.effort (if reasoning supported by model else suppressed)
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
    temperature = kaapi_params.get("temperature")
    instructions = kaapi_params.get("instructions")
    knowledge_base_ids = kaapi_params.get("knowledge_base_ids")
    max_num_results = kaapi_params.get("max_num_results")

    support_reasoning = litellm.supports_reasoning(model=f"openai/{model}")

    # Handle reasoning vs temperature mutual exclusivity
    if support_reasoning:
        if reasoning is not None:
            openai_params["reasoning"] = {"effort": reasoning}

        if temperature is not None:
            warnings.append(
                "Parameter 'temperature' was suppressed because the selected model "
                "supports reasoning, and temperature is ignored when reasoning is enabled."
            )
    else:
        if reasoning is not None:
            warnings.append(
                "Parameter 'reasoning' was suppressed because the selected model "
                "does not support reasoning."
            )

        if temperature is not None:
            openai_params["temperature"] = temperature

    if model:
        openai_params["model"] = model

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


def map_kaapi_to_google_params(kaapi_params: dict) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to Google AI (Gemini) API parameters.

    This mapper transforms standardized Kaapi parameters into Google-specific
    parameter format for the Gemini API.

    Args:
        kaapi_params: Dictionary with standardized Kaapi parameters

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

    # Model is present in all param types
    model = kaapi_params.get("model")
    if not model:
        return {}, ["Missing required 'model' parameter"]

    google_params["model"] = kaapi_params.get("model")

    # Instructions for STT prompts
    instructions = kaapi_params.get("instructions")
    if instructions:
        google_params["instructions"] = instructions

    temperature = kaapi_params.get("temperature")

    if temperature is not None:
        google_params["temperature"] = temperature

    # TTS Config
    voice = kaapi_params.get("voice")
    if voice:
        google_params["voice"] = voice

    language = kaapi_params.get("language")
    if language:
        google_params["language"] = language

    response_format = kaapi_params.get("response_format")
    if response_format:
        google_params["response_format"] = response_format

    reasoning = kaapi_params.get("reasoning")
    if reasoning:
        google_params["reasoning"] = reasoning

    # Warn about unsupported parameters
    if kaapi_params.get("knowledge_base_ids"):
        # TODO: Will take up later, when we add google filesearch tool support
        warnings.append(
            "Parameter 'knowledge_base_ids' is not supported by Google AI and was ignored."
        )

    return google_params, warnings


def map_kaapi_to_sarvam_params(kaapi_params: dict) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to SarvamAI API parameters.

    Handles both STTLLMParams and TTSLLMParams.

    STTLLMParams: model, instructions, input_language, output_language, response_format, temperature
    TTSLLMParams: model, voice, language, response_format

    Args:
        kaapi_params: Dictionary with standardized Kaapi parameters

    Returns:
        Tuple of:
        - Dictionary of SarvamAI API parameters
        - List of warnings for unsupported parameters
    """
    sarvam_params = {}
    warnings = []

    # Model is required for all completion types
    model = kaapi_params.get("model")
    if not model:
        return {}, ["Missing required 'model' parameter"]
    sarvam_params["model"] = model

    # Determine if STT or TTS based on presence of specific params
    voice = kaapi_params.get("voice")
    input_language = kaapi_params.get("input_language")

    if voice is not None:
        # TTS mode - map TTSLLMParams
        sarvam_params["speaker"] = voice

        language = kaapi_params.get("language")
        if not language:
            return {}, ["Missing required 'language' parameter for TTS"]
        sarvam_params["target_language_code"] = language

        response_format = kaapi_params.get("response_format")
        if response_format:
            # Map audio format to SarvamAI codec
            format_mapping = {"mp3": "mp3", "wav": "wav", "ogg": "ogg"}
            sarvam_params["output_audio_codec"] = format_mapping.get(
                response_format, "wav"
            )

    elif input_language is not None or kaapi_params.get("output_language") is not None:
        # STT mode - map STTLLMParams
        output_language = kaapi_params.get("output_language")
        transcription_mode = "transcribe"

        if input_language == "auto":
            sarvam_params["language_code"] = "unknown"
        elif input_language:
            sarvam_params["language_code"] = input_language

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

    return sarvam_params, warnings


def map_kaapi_to_elevenlabs_params(kaapi_params: dict) -> tuple[dict, list[str]]:
    """
    Map Kaapi-abstracted parameters to ElevenLab API params
    Handles both STTLLMParams and TTSLLMParams.

    STTLLMParams: model, instructions, input_language, output_language, response_format, temperature
    TTSLLMParams: model, voice, language, response_format

    Args:
        kaapi_params: Dictionary with standardized Kaapi parameters

    Returns:
        Tuple of:
        - Dictionary of ELevenlabs API parameters
        - List of warnings for unsupported parameters

    """
    elevenlabs_params = {}
    warnings = []

    model_id = kaapi_params.get("model")
    if not model_id:
        return {}, ["Missing required 'model' parameter"]
    elevenlabs_params["model_id"] = model_id

    # determine if STT or TTS bases on specific params
    voice = kaapi_params.get("voice")
    input_language = kaapi_params.get("input_language")

    if voice is not None:
        # TTS Mode
        voice_id = voice_to_id(voice)
        if not voice_id:
            return {}, [f"Unsupported voice '{voice}' for ElevenLabs TTS"]
        elevenlabs_params["voice_id"] = voice_id
        language = kaapi_params.get("language")
        if not language:
            return {}, ["Missing required 'language' parameter for TTS"]
        elevenlabs_lang = bcp47_to_elevenlabs_lang(language)
        if not elevenlabs_lang:
            return {}, [f"Unsupported language '{language}' for ElevenLabs TTS"]
        elevenlabs_params["language_code"] = elevenlabs_lang

        response_format = kaapi_params.get("response_format")
        if response_format:
            # Map audio format to Elevenlabs codec
            # supports mp3, wav and opus
            format_mapping = {
                "mp3": "mp3_44100_128",
                "wav": "wav_24000",
                "opus": "opus_48000_128",
            }
            elevenlabs_params["output_format"] = format_mapping.get(
                response_format, "mp3_44100_128"
            )
    elif input_language is not None or kaapi_params.get("output_language") is not None:
        # STT mode --> map STTLLMParams
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
    return elevenlabs_params, warnings


def transform_kaapi_config_to_native(
    kaapi_config: KaapiCompletionConfig,
) -> tuple[NativeCompletionConfig, list[str]]:
    """Transform Kaapi completion config to native provider config with mapped parameters.

    Supports OpenAI,Google AI and Sarvam AI providers.

    Args:
        kaapi_config: KaapiCompletionConfig with abstracted parameters

    Returns:
        Tuple of:
        - NativeCompletionConfig with provider-native parameters ready for API
        - List of warnings for suppressed/ignored parameters
    """
    # TODO change from magic string to enums
    if kaapi_config.provider == "openai":
        mapped_params, warnings = map_kaapi_to_openai_params(kaapi_config.params)
        return (
            NativeCompletionConfig(
                provider="openai-native", params=mapped_params, type=kaapi_config.type
            ),
            warnings,
        )

    if kaapi_config.provider == "google":
        mapped_params, warnings = map_kaapi_to_google_params(kaapi_config.params)
        return (
            NativeCompletionConfig(
                provider="google-native", params=mapped_params, type=kaapi_config.type
            ),
            warnings,
        )

    if kaapi_config.provider == "sarvamai":
        mapped_params, warnings = map_kaapi_to_sarvam_params(kaapi_config.params)
        return (
            NativeCompletionConfig(
                provider="sarvamai-native", params=mapped_params, type=kaapi_config.type
            ),
            warnings,
        )

    if kaapi_config.provider == "elevenlabs":
        mapped_params, warnings = map_kaapi_to_elevenlabs_params(kaapi_config.params)

        return (
            NativeCompletionConfig(
                provider="elevenlabs-native",
                params=mapped_params,
                type=kaapi_config.type,
            ),
            warnings,
        )

    raise ValueError(f"Unsupported provider: {kaapi_config.provider}")
