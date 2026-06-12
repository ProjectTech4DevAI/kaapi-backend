from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    GOOGLE = "google"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"
    ANTHROPIC = "anthropic"
    GOOGLE_AISTUDIO = "google-aistudio"
    PROXY = "proxy"


class CompletionType(StrEnum):
    TEXT = "text"
    STT = "stt"
    TTS = "tts"


class Modality(StrEnum):
    TEXT = "TEXT"
    AUDIO = "AUDIO"
    IMAGE = "IMAGE"
    FILES = "FILES"


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
