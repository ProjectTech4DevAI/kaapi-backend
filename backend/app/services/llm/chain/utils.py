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


# Supported BCP-47 language codes for speech-to-speech
# These are the valid values that can be used directly in API requests
SUPPORTED_LANGUAGE_CODES = {
    # Auto-detect
    "auto",  # Auto-detection (maps to "unknown" for Sarvam)
    "unknown",  # Explicit unknown for Sarvam
    # Primary Indian languages (BCP-47 codes)
    "en-IN",  # English
    "hi-IN",  # Hindi (also used for Hinglish/code-switching)
    "bn-IN",  # Bengali
    "kn-IN",  # Kannada
    "ml-IN",  # Malayalam
    "mr-IN",  # Marathi
    "od-IN",  # Odia
    "pa-IN",  # Punjabi
    "ta-IN",  # Tamil
    "te-IN",  # Telugu
    "gu-IN",  # Gujarati
    # Additional languages (saaras:v3)
    "as-IN",  # Assamese
    "ur-IN",  # Urdu
    "ne-IN",  # Nepali
    "kok-IN",  # Konkani
    "ks-IN",  # Kashmiri
    "sd-IN",  # Sindhi
    "sa-IN",  # Sanskrit
    "sat-IN",  # Santali
    "mni-IN",  # Manipuri
    "brx-IN",  # Bodo
    "mai-IN",  # Maithili
    "doi-IN",  # Dogri
}


def build_stt_block(model: STTModel, language_code: str) -> ChainBlock:
    """Build STT (Speech-to-Text) block configuration.

    Args:
        model: STT model enum
        language_code: BCP-47 language code (e.g., "hi-IN", "en-IN") or "auto" for auto-detection

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
        # Map "auto" to "unknown" for Sarvam auto-detection
        params["language_code"] = (
            "unknown" if language_code == "auto" else language_code
        )
        params["mode"] = "transcribe"
    elif provider == "google-native":
        # Google requires specific language code, fallback to en-IN if auto/unknown
        params["language_code"] = (
            "en-IN" if language_code in ("auto", "unknown") else language_code
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


def build_tts_block(model: TTSModel, language_code: str = "en-IN") -> ChainBlock:
    """Build TTS (Text-to-Speech) block configuration.

    Args:
        model: TTS model enum
        language_code: ISO language code (e.g., "hi-IN"), or "{{detected}}" to use language detected by STT

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
        # Use language_code (can be "{{detected}}" marker or actual code)
        params["target_language_code"] = language_code
        params["speaker"] = voice
        params["output_audio_codec"] = "opus"  # WhatsApp compatible
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
