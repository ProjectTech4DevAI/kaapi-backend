from enum import StrEnum
from typing import Literal, Union


class Provider(StrEnum):
    OPENAI = "openai"
    GOOGLE = "google"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"
    ANTHROPIC = "anthropic"
    GOOGLE_AISTUDIO = "google-aistudio"
    GOOGLE_GCP = "google-gcp"
    PROXY = "proxy"


# Provider Literals reused across request models. Reference Provider enum
# members directly so a rename in the enum surfaces at every call site
# instead of leaving behind stale magic strings.
STTProvider = Literal[
    Provider.GOOGLE,
    Provider.GOOGLE_GCP,
    Provider.SARVAMAI,
    Provider.ELEVENLABS,
    Provider.GOOGLE_AISTUDIO,
]
TTSProvider = Literal[
    Provider.GOOGLE,
    Provider.GOOGLE_GCP,
    Provider.SARVAMAI,
    Provider.ELEVENLABS,
    Provider.GOOGLE_AISTUDIO,
]
RAGProvider = Literal[Provider.OPENAI, Provider.GOOGLE_AISTUDIO]

TextProvider = Literal[
    Provider.OPENAI,
    Provider.GOOGLE,
    Provider.ANTHROPIC,
    Provider.GOOGLE_AISTUDIO,
    Provider.GOOGLE_GCP,
]

KaapiProvider = Union[TextProvider, STTProvider, TTSProvider]

GoogleProvider = Literal[Provider.GOOGLE, Provider.GOOGLE_AISTUDIO]

# Native provider names are the Kaapi providers with a "-native" suffix.
# Kept as explicit strings since there's no corresponding enum member.
NativeProvider = Literal[
    "openai-native",
    "google-native",
    "sarvamai-native",
    "elevenlabs-native",
    "anthropic-native",
    "google-aistudio-native",
    "google-gcp-native",
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
# TTS output). This Literal is the single source of truth; `SUPPORTED_LANGUAGE_CODES`
# in `app/services/llm/chain/utils.py` is derived from it via `get_args`.
SUPPORTED_STS_LANGUAGE_CODES = Literal[
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

# Aliases accepted for STT/TTS/STS language fields — bare ISO 639 codes and
# English language names, all lowercase — mapped to the canonical BCP-47 tag.
LANGUAGE_ALIAS_TO_BCP47: dict[str, str] = {
    "en": "en-IN",
    "english": "en-IN",
    "hi": "hi-IN",
    "hindi": "hi-IN",
    "bn": "bn-IN",
    "bengali": "bn-IN",
    "kn": "kn-IN",
    "kannada": "kn-IN",
    "ml": "ml-IN",
    "malayalam": "ml-IN",
    "mr": "mr-IN",
    "marathi": "mr-IN",
    "od": "od-IN",
    "or": "od-IN",
    "odia": "od-IN",
    "oriya": "od-IN",
    "pa": "pa-IN",
    "punjabi": "pa-IN",
    "ta": "ta-IN",
    "tamil": "ta-IN",
    "te": "te-IN",
    "telugu": "te-IN",
    "gu": "gu-IN",
    "gujarati": "gu-IN",
    "as": "as-IN",
    "assamese": "as-IN",
    "ur": "ur-IN",
    "urdu": "ur-IN",
    "ne": "ne-IN",
    "nepali": "ne-IN",
    "kok": "kok-IN",
    "konkani": "kok-IN",
    "ks": "ks-IN",
    "kashmiri": "ks-IN",
    "sd": "sd-IN",
    "sindhi": "sd-IN",
    "sa": "sa-IN",
    "sanskrit": "sa-IN",
    "sat": "sat-IN",
    "santali": "sat-IN",
    "mni": "mni-IN",
    "manipuri": "mni-IN",
    "meitei": "mni-IN",
    "brx": "brx-IN",
    "bodo": "brx-IN",
    "mai": "mai-IN",
    "maithili": "mai-IN",
    "doi": "doi-IN",
    "dogri": "doi-IN",
}


def normalize_bcp47_language(value: str) -> str:
    """Best-effort normalize a user-supplied language value (English name,
    bare ISO 639 code, or BCP-47 tag, any casing) to the canonical Kaapi
    BCP-47 tag, e.g. 'hindi' / 'HI' / 'hi-in' -> 'hi-IN'.

    Unrecognized input is returned unchanged so callers keep validating/
    rejecting it themselves.
    """
    key = value.strip().lower()
    if key in ("auto", "unknown"):
        return key
    if key in LANGUAGE_ALIAS_TO_BCP47:
        return LANGUAGE_ALIAS_TO_BCP47[key]
    parts = key.split("-")
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1].upper()}"
    return value


DEFAULT_STT_MODEL = "gemini-2.5-pro"
DEFAULT_TTS_MODEL = "gemini-3.1-flash-tts-preview"
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
