from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    GOOGLE = "google"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"


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
