"""Shared constants for TTS evaluation services.

Kept free of `app.models` imports: `app.models.tts_evaluation` imports this module
for its request validator, so importing models here would create a cycle.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.core.providers import Provider


class TTSExecutionModeEnum(StrEnum):
    """How a TTS model's synthesis is executed.

    BATCH goes through a provider Batch API polled by the cron; SYNC is called
    directly from chunked Celery tasks.
    """

    BATCH = "batch"
    SYNC = "sync"


@dataclass(frozen=True)
class TTSModelSpec:
    provider: Provider
    execution_mode: TTSExecutionModeEnum
    default_voice: str


GEMINI_PRO_TTS_MODEL = "gemini-2.5-pro-preview-tts"
SARVAM_BULBUL_V3_MODEL = "bulbul:v3"
ELEVENLABS_V3_MODEL = "eleven_v3"
ELEVENLABS_V4_MODEL = "eleven_v4"

# Default voice configuration (Gemini prebuilt voice)
DEFAULT_VOICE_NAME = "Kore"
SARVAM_DEFAULT_SPEAKER = "shubh"
# "Rahul S - Hindi Conversational": closest match to Sarvam's male speaker so the
# providers are roughly comparable in a side-by-side review.
ELEVENLABS_DEFAULT_VOICE_ID = "2cdvnKJ5TZi631y5PN1s"

TTS_MODEL_REGISTRY: dict[str, TTSModelSpec] = {
    GEMINI_PRO_TTS_MODEL: TTSModelSpec(
        provider=Provider.GOOGLE_AISTUDIO,
        execution_mode=TTSExecutionModeEnum.BATCH,
        default_voice=DEFAULT_VOICE_NAME,
    ),
    SARVAM_BULBUL_V3_MODEL: TTSModelSpec(
        provider=Provider.SARVAMAI,
        execution_mode=TTSExecutionModeEnum.SYNC,
        default_voice=SARVAM_DEFAULT_SPEAKER,
    ),
    ELEVENLABS_V3_MODEL: TTSModelSpec(
        provider=Provider.ELEVENLABS,
        execution_mode=TTSExecutionModeEnum.SYNC,
        default_voice=ELEVENLABS_DEFAULT_VOICE_ID,
    ),
    ELEVENLABS_V4_MODEL: TTSModelSpec(
        provider=Provider.ELEVENLABS,
        execution_mode=TTSExecutionModeEnum.SYNC,
        default_voice=ELEVENLABS_DEFAULT_VOICE_ID,
    ),
}

# Supported TTS models for evaluation
SUPPORTED_TTS_MODELS = list(TTS_MODEL_REGISTRY.keys())

# Default TTS model
DEFAULT_TTS_MODEL = GEMINI_PRO_TTS_MODEL

# Default style prompt for TTS synthesis
DEFAULT_STYLE_PROMPT = "Read in a calm, professional customer service tone"

# BCP-47 tag used when the dataset has no language configured.
DEFAULT_TTS_LANGUAGE_CODE = "en-IN"

# Every stored evaluation clip is normalized to this format so providers compare 1:1.
TTS_SAMPLE_RATE_HZ = 24000
TTS_SAMPLE_WIDTH_BYTES = 2
TTS_CHANNELS = 1

TTS_AUDIO_SUBDIRECTORY = "evaluations/tts/audio"
TTS_AUDIO_CONTENT_TYPE = "audio/wav"
TTS_AUDIO_FILE_EXTENSION = "wav"

SARVAM_PACE = 1.0
ELEVENLABS_OUTPUT_FORMAT = f"pcm_{TTS_SAMPLE_RATE_HZ}"

# 25 items x 4 workers keeps a chunk well inside CELERY_TASK_SOFT_TIME_LIMIT
# even with retries on slow (~10s) ElevenLabs v3 calls.
TTS_SYNC_CHUNK_SIZE = 25
TTS_SYNC_MAX_WORKERS = 4
TTS_SYNC_REQUEST_TIMEOUT_SECONDS = 60
TTS_SYNC_MAX_ATTEMPTS = 3
TTS_SYNC_BACKOFF_BASE_SECONDS = 1.0
TTS_SYNC_BACKOFF_MAX_SECONDS = 20.0
TTS_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})

# Source tag for failures that are Kaapi-side (see error-handling convention).
KAAPI_TAG = "KAAPI"

# Commit Gemini batch-result writes in slices so a timeout keeps earlier progress.
TTS_RESULT_WRITE_CHUNK_SIZE = 50


def get_tts_model_spec(model: str) -> TTSModelSpec:
    """Return the registry spec for a model; raises KeyError for unknown models."""
    return TTS_MODEL_REGISTRY[model]


def split_models_by_execution_mode(
    models: list[str],
) -> tuple[list[str], list[str]]:
    """Split requested models into (batch_models, sync_models), preserving order."""
    batch_models: list[str] = []
    sync_models: list[str] = []
    for model in models:
        spec = get_tts_model_spec(model)
        if spec.execution_mode == TTSExecutionModeEnum.BATCH:
            batch_models.append(model)
        else:
            sync_models.append(model)
    return batch_models, sync_models
