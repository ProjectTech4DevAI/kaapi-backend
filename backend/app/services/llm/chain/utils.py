"""Utility functions for LLM chain operations, including speech-to-speech helpers."""

from typing import Any, Literal

from app.models.llm.request import (
    ChainBlock,
    ConfigBlob,
    KaapiCompletionConfig,
    LLMCallConfig,
    LLMModel,
    NativeCompletionConfig,
    STTModel,
    TextLLMParams,
    TTSModel,
)


# Supported languages for speech-to-speech (BCP-47 language codes)
LANGUAGE_CODES = {
    # Auto-detect
    "auto": "unknown",  # Sarvam auto-detection
    # Primary Indian languages
    "english": "en-IN",
    "hindi": "hi-IN",
    "hinglish": "hi-IN",  # Code-switching, treat as Hindi
    "bengali": "bn-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "odia": "od-IN",
    "punjabi": "pa-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "gujarati": "gu-IN",
    # Additional languages (saaras:v3)
    "assamese": "as-IN",
    "urdu": "ur-IN",
    "nepali": "ne-IN",
    "konkani": "kok-IN",
    "kashmiri": "ks-IN",
    "sindhi": "sd-IN",
    "sanskrit": "sa-IN",
    "santali": "sat-IN",
    "manipuri": "mni-IN",
    "bodo": "brx-IN",
    "maithili": "mai-IN",
    "dogri": "doi-IN",
}


def get_language_code(language: str | None, default: str = "auto") -> str:
    """Convert language name to BCP-47 language code.

    Args:
        language: Language name (e.g., "hindi", "english", "auto")
        default: Default language if not specified (default: "auto")

    Returns:
        BCP-47 language code (e.g., "hi-IN", "en-IN", "unknown" for auto-detect)
    """
    lang = (language or default).lower()
    return LANGUAGE_CODES.get(lang, LANGUAGE_CODES["auto"])


def build_stt_block(model: STTModel, language_code: str) -> ChainBlock:
    """Build STT (Speech-to-Text) block configuration.

    Args:
        model: STT model enum
        language_code: ISO language code (e.g., "hi-IN")

    Returns:
        ChainBlock configured for STT
    """
    # Map model to provider and actual model name
    model_configs: dict[
        STTModel,
        tuple[Literal["sarvamai-native", "google-native", "openai-native"], str],
    ] = {
        STTModel.SARVAM: ("sarvamai-native", "saaras:v3"),
        STTModel.GEMINI_PRO: ("google-native", "gemini-2.5-pro"),
    }

    provider, model_name = model_configs[model]

    # Build native config (provider-specific params)
    params: dict[str, Any] = {
        "model": model_name,
    }

    # Add provider-specific parameters
    if provider == "sarvamai-native":
        # Use "unknown" for automatic language detection, or specific BCP-47 code
        params["language_code"] = (
            language_code if language_code != "unknown" else "unknown"
        )
        params["mode"] = "transcription"
    elif provider == "google-native":
        # Google requires specific language code, fallback to en-IN if unknown
        params["language_code"] = (
            language_code if language_code != "unknown" else "en-IN"
        )

    return ChainBlock(
        config=LLMCallConfig(
            blob=ConfigBlob(
                completion=NativeCompletionConfig(
                    provider=provider,
                    type="stt",
                    params=params,
                )
            )
        ),
        intermediate_callback=True,  # Send STT result to user
        include_provider_raw_response=False,
    )


def build_rag_block(model: LLMModel, knowledge_base_ids: list[str]) -> ChainBlock:
    """Build RAG (Retrieval-Augmented Generation) block configuration.

    Args:
        model: LLM model enum
        knowledge_base_ids: List of knowledge base IDs for retrieval

    Returns:
        ChainBlock configured for RAG
    """
    return ChainBlock(
        config=LLMCallConfig(
            blob=ConfigBlob(
                completion=KaapiCompletionConfig(
                    provider="openai",
                    type="text",
                    params=TextLLMParams(
                        model=model.value,
                        knowledge_base_ids=knowledge_base_ids,
                        temperature=0.1,
                        instructions="Answer the user's question using the provided knowledge base. Be concise and accurate.",
                    ).model_dump(exclude_none=True),
                )
            )
        ),
        intermediate_callback=True,  # Send LLM result to user
        include_provider_raw_response=False,
    )


def build_tts_block(model: TTSModel, language_code: str) -> ChainBlock:
    """Build TTS (Text-to-Speech) block configuration.

    Args:
        model: TTS model enum
        language_code: ISO language code (e.g., "hi-IN")

    Returns:
        ChainBlock configured for TTS
    """
    # Map model to provider and actual model name + voice
    model_configs: dict[
        TTSModel,
        tuple[Literal["sarvamai-native", "google-native", "openai-native"], str, str],
    ] = {
        TTSModel.SARVAM: ("sarvamai-native", "bulbul:v3", "simran"),
        TTSModel.GEMINI_PRO: ("google-native", "gemini-2.5-pro", "default"),
    }

    provider, model_name, voice = model_configs[model]

    # Build native config
    params: dict[str, Any] = {
        "model": model_name,
        "voice": voice,
    }

    # Add provider-specific parameters
    if provider == "sarvamai-native":
        params["target_language_code"] = language_code
        params["speaker"] = voice
        params["output_audio_codec"] = "mp3"  # WhatsApp compatible
    elif provider == "google-native":
        params["language_code"] = language_code
        params["audio_encoding"] = "OGG_OPUS"  # WhatsApp compatible

    return ChainBlock(
        config=LLMCallConfig(
            blob=ConfigBlob(
                completion=NativeCompletionConfig(
                    provider=provider,
                    type="tts",
                    params=params,
                )
            )
        ),
        intermediate_callback=False,  # Final result only
        include_provider_raw_response=False,
    )
