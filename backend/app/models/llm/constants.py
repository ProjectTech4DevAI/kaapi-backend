DEFAULT_STT_MODEL = "gemini-2.5-pro"
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_TTS_VOICE = "Kore"

SUPPORTED_MODELS = {
    ("google", "stt"): [
        DEFAULT_STT_MODEL,
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    ],
    ("google", "tts"): [DEFAULT_TTS_MODEL, "gemini-2.5-pro-preview-tts"],
    ("sarvamai", "stt"): ["saaras:v3"],
    ("sarvamai", "tts"): ["bulbul:v3"],
    ("elevenlabs", "stt"): ["scribe_v2"],
    ("elevenlabs", "tts"): ["eleven_v3"],
    ("openai", "text"): [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5.4",
        "gpt-5.1",
        "gpt-5-mini",
        "gpt-5-nano",
        "o1",
        "o1-preview",
        "o1-mini",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
}

SUPPORTED_VOICES = {
    ("google", "tts"): ["Kore", "Orus", "Leda", "Charon"],
    ("sarvamai", "tts"): ["simran", "shubh", "roopa"],
    ("elevenlabs", "tts"): ["Sarah", "George", "Callum", "Liam"],
}

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
