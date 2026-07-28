from enum import StrEnum
from typing import Literal


class Provider(StrEnum):
    OPENAI = "openai"
    GOOGLE = "google"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"
    ANTHROPIC = "anthropic"
    GOOGLE_AISTUDIO = "google-aistudio"
    PROXY = "proxy"


# Provider Literals reused across request models. Reference Provider enum
# members directly so a rename in the enum surfaces at every call site
# instead of leaving behind stale magic strings.
STTProvider = Literal[
    Provider.GOOGLE,
    Provider.SARVAMAI,
    Provider.ELEVENLABS,
    Provider.GOOGLE_AISTUDIO,
]
TTSProvider = Literal[
    Provider.GOOGLE,
    Provider.SARVAMAI,
    Provider.ELEVENLABS,
    Provider.GOOGLE_AISTUDIO,
]
RAGProvider = Literal[Provider.OPENAI, Provider.GOOGLE_AISTUDIO]

KaapiProvider = Literal[
    Provider.OPENAI,
    Provider.GOOGLE,
    Provider.SARVAMAI,
    Provider.ELEVENLABS,
    Provider.ANTHROPIC,
    Provider.GOOGLE_AISTUDIO,
]

# Native provider names are the Kaapi providers with a "-native" suffix.
# Kept as explicit strings since there's no corresponding enum member.
NativeProvider = Literal[
    "openai-native",
    "google-native",
    "sarvamai-native",
    "elevenlabs-native",
    "anthropic-native",
    "google-aistudio-native",
]


class CompletionType(StrEnum):
    TEXT = "text"
    STT = "stt"
    TTS = "tts"


class Modality(StrEnum):
    TEXT = "TEXT"
    AUDIO = "AUDIO"
    IMAGE = "IMAGE"
    FILES = "FILES"


# BCP-47 language codes accepted by the speech-to-speech endpoint (STT input /
# TTS output). Single source of truth: `SUPPORTED_LANGUAGE_CODES` in
# `app/services/llm/chain/utils.py` derives from this via `get_args`.
STSLanguageCode = Literal[
    "auto",
    "unknown",
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
]


DEFAULT_STT_MODEL = "gemini-2.5-pro"
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_TTS_VOICE = "Kore"
DEFAULT_RAG_MODEL = "gpt-4o"

# Default text-completion model per provider. Used by both the native flow
# (provider.execute) and the Kaapi mapper so the two stay in sync.
DEFAULT_TEXT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4.1-mini",
    "google": "gemini-2.5-pro",
}

DEFAULT_ANTHROPIC_MAX_TOKENS = 4096

DEFAULT_ASSESSMENT_BATCH_MAX_TOKENS = 16384

# Provider-native STT/TTS defaults (used when caller omits model).
DEFAULT_SARVAM_STT_MODEL = "saaras:v3"
DEFAULT_SARVAM_TTS_MODEL = "bulbul:v3"
DEFAULT_ELEVENLABS_STT_MODEL = "scribe_v2"
DEFAULT_ELEVENLABS_TTS_MODEL = "eleven_v3"

# BCP-47 to language tag -> Gemini ISO 639-1 code (Indic + English)
BCP47_LOCALE_TO_GEMINI_LANG: dict[str, str] = {
    "en-IN": "en",
    "hi-IN": "hi",
    "bn-IN": "bn",
    "ta-IN": "ta",
    "te-IN": "te",
    "mr-IN": "mr",
    "gu-IN": "gu",
    "kn-IN": "kn",
    "ml-IN": "ml",
    "pa-IN": "pa",
    "od-IN": "or",
    # "as-IN": "as", //not supported by Gemini
    "sd-IN": "sd",
}

# BCP-47 language tag → ElevenLabs ISO 639-1 code (Indic + English)
BCP47_TO_ELEVENLABS_LANG: dict[str, str] = {
    "en-IN": "en",
    "hi-IN": "hi",
    "bn-IN": "bn",  # Bengali
    "ta-IN": "ta",
    "te-IN": "te",
    "mr-IN": "mr",
    "gu-IN": "gu",
    "kn-IN": "kn",
    "ml-IN": "ml",
    "pa-IN": "pa",
    # "od-IN": "or",  # Not supported by Elevenlabs explicitly but works in auto detect mode
    "as-IN": "as",
    "ur-IN": "ur",
    "ne-IN": "ne",
    "sd-IN": "sd",
}

ELEVENLABS_VOICE_TO_ID: dict[str, str] = {
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Callum": "N2lVS1w4EtoT3dr4eOWO",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
}
