"""Shared constants for TTS evaluation services.

Kept free of `app.models` imports: `app.models.tts_evaluation` imports this module,
so importing models here would create a cycle.
"""

# Gemini models go through the Gemini Batch API; Sarvam/ElevenLabs are called directly.
GEMINI_TTS_MODELS = ["gemini-2.5-pro-preview-tts", "gemini-3.1-flash-tts-preview"]
SARVAM_TTS_MODELS = ["bulbul:v3"]
ELEVENLABS_TTS_MODELS = ["eleven_v4", "eleven_v3"]

SYNC_TTS_MODELS = SARVAM_TTS_MODELS + ELEVENLABS_TTS_MODELS

# Supported TTS models for evaluation
SUPPORTED_TTS_MODELS = GEMINI_TTS_MODELS + SYNC_TTS_MODELS

# Default TTS model
DEFAULT_TTS_MODEL = "gemini-2.5-pro-preview-tts"

# Default voice configuration
DEFAULT_VOICE_NAME = "Kore"
SARVAM_SPEAKER = "shubh"
# "Rahul S - Hindi Conversational": closest match to Sarvam's male speaker for side-by-side review.
ELEVENLABS_VOICE_ID = "2cdvnKJ5TZi631y5PN1s"

# Default style prompt for TTS synthesis
DEFAULT_STYLE_PROMPT = "Read in a calm, professional customer service tone"

DEFAULT_TTS_LANGUAGE = "en-IN"

# Dataset locale (ISO 639-1) -> BCP-47 tag; Sarvam spells Odia "od-IN", not "or-IN".
LOCALE_TO_BCP47 = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "or": "od-IN",
}

# Rows synthesized at once per sync task; kept low because eleven_v3 and eleven_v4
# share one ElevenLabs key and its concurrent-request limit.
TTS_SYNC_MAX_WORKERS = 2

TTS_AUDIO_SUBDIRECTORY = "evaluations/tts/audio"

# Once a run is older than the Celery hard time limit plus this grace, no sync worker can
# still be writing its rows, so the cron fails any that are left PENDING.
TTS_SYNC_STALE_GRACE_SECONDS = 600
