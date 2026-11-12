"""Allowed and default voices per TTS evaluation model.

Separate from `constants.py` because the ElevenLabs voice names come from
`app.models.llm.constants`, which `constants.py` must not import.
"""

from app.models.llm.constants import ELEVENLABS_VOICE_TO_ID
from app.services.tts_evaluations.constants import (
    DEFAULT_ELEVENLABS_VOICE,
    DEFAULT_VOICE_NAME,
    ELEVENLABS_TTS_MODELS,
    GEMINI_BATCH_TTS_MODELS,
    GEMINI_TTS_VOICES,
    SARVAM_SPEAKER,
    SARVAM_TTS_MODELS,
    SARVAM_TTS_VOICES,
)


def voices_for_model(model: str) -> list[str]:
    """Return the voices a TTS evaluation model accepts; empty for unknown models."""
    if model in GEMINI_BATCH_TTS_MODELS:
        return list(GEMINI_TTS_VOICES)
    if model in SARVAM_TTS_MODELS:
        return list(SARVAM_TTS_VOICES)
    if model in ELEVENLABS_TTS_MODELS:
        return list(ELEVENLABS_VOICE_TO_ID.keys())
    return []


def default_voice_for_model(model: str) -> str:
    """Return the voice used when a request (or a pre-voice result row) has none."""
    if model in SARVAM_TTS_MODELS:
        return SARVAM_SPEAKER
    if model in ELEVENLABS_TTS_MODELS:
        return DEFAULT_ELEVENLABS_VOICE
    return DEFAULT_VOICE_NAME


def resolve_voice(model: str, voice: str | None) -> str:
    """Validate a requested voice for `model` and return it in the provider's casing.

    Raises:
        ValueError: If the voice is not in the model's allowed set.
    """
    if voice is None:
        return default_voice_for_model(model)

    allowed = voices_for_model(model)
    # Sarvam speaker IDs are lowercase-only, so client casing is normalised for it.
    candidate = voice.lower() if model in SARVAM_TTS_MODELS else voice
    if candidate not in allowed:
        raise ValueError(
            f"Unsupported voice '{voice}' for model '{model}'. "
            f"Allowed voices are: {', '.join(allowed)}"
        )
    return candidate
