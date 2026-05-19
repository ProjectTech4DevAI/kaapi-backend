"""Utility functions for LLM chain operations, including speech-to-speech helpers."""

from typing import Literal

from app.models.llm.request import (
    ChainBlock,
    ConfigBlob,
    KaapiCompletionConfig,
    LLMCallConfig,
    STTLLMParams,
    TextLLMParams,
    TTSLLMParams,
)

KaapiProvider = Literal["openai", "google", "sarvamai", "elevenlabs"]

# BCP-47 language codes accepted by the speech-to-speech endpoint.
SUPPORTED_LANGUAGE_CODES = {
    "auto",
    "unknown",
    # Primary Indian languages
    "en-IN",
    "hi-IN",
    "bn-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "gu-IN",
    # Additional languages
    "as-IN",
    "ur-IN",
    "ne-IN",
    "kok-IN",
    "ks-IN",
    "sd-IN",
    "sa-IN",
    "sat-IN",
    "mni-IN",
    "brx-IN",
    "mai-IN",
    "doi-IN",
}

DEFAULT_RAG_INSTRUCTIONS = (
    "Answer the user's question using the provided knowledge base. "
    "Be concise and accurate."
)


def _kaapi_block(
    provider: KaapiProvider | None,
    type_: Literal["text", "stt", "tts"],
    params: dict,
    *,
    intermediate_callback: bool,
) -> ChainBlock:
    """Wrap a validated params dict in a Kaapi chain block.

    Provider translation (Sarvam/Google/ElevenLabs/OpenAI) is handled by the
    mapper layer; this builder only constructs the typed wrapper.
    """
    return ChainBlock(
        config=LLMCallConfig(
            blob=ConfigBlob(
                completion=KaapiCompletionConfig(
                    provider=provider,
                    type=type_,
                    params=params,
                )
            )
        ),
        intermediate_callback=intermediate_callback,
        include_provider_raw_response=False,
    )


def build_stt_block(
    params: STTLLMParams,
    provider: KaapiProvider | None = None,
) -> ChainBlock:
    """STT block. Provider defaults to 'google' via KaapiCompletionConfig."""
    return _kaapi_block(
        provider,
        "stt",
        params.model_dump(exclude_none=True),
        intermediate_callback=True,
    )


def build_rag_block(
    params: TextLLMParams,
    knowledge_base_ids: list[str],
    provider: KaapiProvider = "openai",
) -> ChainBlock:
    """RAG block. Injects knowledge_base_ids and a default instruction if absent."""
    merged = params.model_copy(
        update={
            "knowledge_base_ids": knowledge_base_ids,
            "instructions": params.instructions or DEFAULT_RAG_INSTRUCTIONS,
        }
    )
    return _kaapi_block(
        provider,
        "text",
        merged.model_dump(exclude_none=True),
        intermediate_callback=True,
    )


def build_tts_block(
    params: TTSLLMParams,
    provider: KaapiProvider | None = None,
) -> ChainBlock:
    """TTS block. Provider defaults to 'google' via KaapiCompletionConfig."""
    return _kaapi_block(
        provider,
        "tts",
        params.model_dump(exclude_none=True),
        intermediate_callback=False,
    )
