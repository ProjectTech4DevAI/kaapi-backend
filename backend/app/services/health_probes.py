import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import redis
from sentry_sdk.crons import MonitorStatus, capture_checkin
from sentry_sdk.types import MonitorConfig
from sqlmodel import Session

from app.core.config import settings
from app.core.db import engine
from app.crud.jobs import JobCrud
from app.models.job import JobStatus
from app.models.llm.constants import CompletionType
from app.models.llm.request import (
    AudioContent,
    AudioInput,
    ConfigBlob,
    KaapiCompletionConfig,
    LLMCallConfig,
    LLMCallRequest,
    QueryParams,
)
from app.services.llm.jobs import start_job

logger = logging.getLogger(__name__)

_PROBE_INPUT = "ping"

# ~1sec "Hello" clip used as the STT probe's input audio.
_STT_AUDIO_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "health_probe_hello.ogg"
)
_STT_AUDIO_MIME = "audio/ogg"
_stt_audio_b64: str | None = None

HEALTH_PROBES_MONITOR_SLUG = "health-probes-cron-job"
HEALTH_PROBES_MONITOR_CONFIG: MonitorConfig = {
    "schedule": {
        "type": "interval",
        "value": settings.HEALTH_PROBE_INTERVAL_MINUTES,
        "unit": "minute",
    },
    "timezone": "UTC",
    "checkin_margin": 2,
    "max_runtime": 2 * settings.HEALTH_PROBE_INTERVAL_MINUTES,
    "failure_issue_threshold": 2,
    "recovery_threshold": 1,
}

# Rotation state lives in Redis, not a table — no CRUD, no tenant isolation needed.
_LAST_JOB_ID_KEY = "health_probe:last_job_id"
_INDEX_KEY = "health_probe:index"

_redis_client: redis.Redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

Modality = Literal["text", "tts", "stt"]


@dataclass(frozen=True)
class Probe:
    provider: str
    model: str
    modality: Modality
    # Extra Kaapi params a provider's mapper requires beyond `model`
    # (Sarvam TTS needs `language`, ElevenLabs TTS needs `voice`).
    params: dict[str, Any] = field(default_factory=dict)


_PROBES: list[Probe] = [
    # Text
    Probe(provider="openai", model="gpt-4o-mini", modality="text"),
    Probe(provider="google", model="gemini-2.5-flash", modality="text"),
    Probe(provider="google", model="gemini-2.5-pro", modality="text"),
    Probe(provider="anthropic", model="claude-sonnet-4-6", modality="text"),
    # TTS
    Probe(provider="google", model="gemini-2.5-flash-preview-tts", modality="tts"),
    Probe(provider="google", model="gemini-3.1-flash-tts-preview", modality="tts"),
    Probe(provider="google", model="gemini-2.5-pro-preview-tts", modality="tts"),
    Probe(
        provider="sarvamai",
        model="bulbul:v3",
        modality="tts",
        params={"language": "en-IN"},
    ),
    Probe(
        provider="elevenlabs",
        model="eleven_v3",
        modality="tts",
        params={"voice": "Sarah"},
    ),
    # STT
    Probe(provider="google", model="gemini-2.5-pro", modality="stt"),
    Probe(provider="google", model="gemini-2.5-flash", modality="stt"),
    Probe(provider="google", model="gemini-3.1-pro-preview", modality="stt"),
    Probe(provider="sarvamai", model="saaras:v3", modality="stt"),
    Probe(provider="elevenlabs", model="scribe_v2", modality="stt"),
]


def _load_stt_audio_b64() -> str:
    global _stt_audio_b64
    if _stt_audio_b64 is None:
        with open(_STT_AUDIO_PATH, "rb") as f:
            _stt_audio_b64 = base64.b64encode(f.read()).decode("ascii")
    return _stt_audio_b64


def _build_probe_request(probe: Probe, index: int) -> LLMCallRequest:
    completion_type = {
        "text": CompletionType.TEXT,
        "tts": CompletionType.TTS,
        "stt": CompletionType.STT,
    }[probe.modality]

    kaapi_config = KaapiCompletionConfig(
        provider=probe.provider,
        type=completion_type,
        params={"model": probe.model, **probe.params},
    )

    if probe.modality == "stt":
        query_input: str | AudioInput = AudioInput(
            content=AudioContent(
                format="base64",
                value=_load_stt_audio_b64(),
                mime_type=_STT_AUDIO_MIME,
            )
        )
    else:
        query_input = _PROBE_INPUT

    # No callback_url: the probe reads its result by polling `Job.status` on
    # the next tick, so callback delivery (a best-effort side channel that
    # never affects Job.status) buys nothing here.
    return LLMCallRequest(
        query=QueryParams(input=query_input),
        config=LLMCallConfig(blob=ConfigBlob(completion=kaapi_config)),
        request_metadata={
            "health_probe": True,
            "probe_index": index,
            "provider": probe.provider,
            "modality": probe.modality,
            "model": probe.model,
        },
    )


def _check_previous_probe(project_id: int) -> JobStatus | None:
    # Missing key (first run, Redis eviction) means skip the check-in, not fail the tick.
    try:
        last_job_id = _redis_client.get(_LAST_JOB_ID_KEY)
    except redis.RedisError as e:
        logger.warning(f"[_check_previous_probe] Redis GET failed | error: {e}")
        return None

    if last_job_id is None:
        logger.warning(
            f"[_check_previous_probe] {_LAST_JOB_ID_KEY} missing — skipping check-in"
        )
        return None

    try:
        job_uuid = UUID(str(last_job_id))
    except ValueError as e:
        logger.warning(
            f"[_check_previous_probe] Malformed job id | value: {last_job_id!r}, error: {e}"
        )
        return None

    with Session(engine) as session:
        job = JobCrud(session=session).get(job_id=job_uuid, project_id=project_id)

    if job is None:
        logger.warning(f"[_check_previous_probe] Job not found | job_id: {last_job_id}")
        return None

    return job.status


def _capture_previous_result_checkin(previous_status: JobStatus | None) -> None:
    if previous_status is None:
        return
    # SUCCESS means the probe actually ran and returned; FAILED or a
    # still-PENDING/PROCESSING job both mean a real failure.
    monitor_status = (
        MonitorStatus.OK
        if previous_status == JobStatus.SUCCESS
        else MonitorStatus.ERROR
    )
    capture_checkin(
        monitor_slug=HEALTH_PROBES_MONITOR_SLUG,
        status=monitor_status,
        monitor_config=HEALTH_PROBES_MONITOR_CONFIG,
    )


def _claim_next_probe_index() -> int:
    # Redis INCR is atomic, so two overlapping ticks can never claim the same
    # slot (which would fire the same probe twice while skipping another).
    try:
        count = _redis_client.incr(_INDEX_KEY)
    except redis.RedisError as e:
        logger.warning(f"[_claim_next_probe_index] Redis INCR failed | error: {e}")
        return 0

    try:
        return (int(count) - 1) % len(_PROBES)
    except (TypeError, ValueError):
        logger.warning(f"[_claim_next_probe_index] Non-integer index | raw: {count!r}")
        return 0


def _store_last_job_id(job_id: UUID) -> None:
    try:
        _redis_client.set(_LAST_JOB_ID_KEY, str(job_id))
    except redis.RedisError as e:
        logger.error(
            f"[_store_last_job_id] Redis SET failed | job_id: {job_id}, error: {e}"
        )


def run_health_probe_tick() -> dict[str, Any]:
    """Check the previous probe's result, then fire the next one round-robin."""
    org_id = settings.HEALTH_PROBE_ORG_ID
    project_id = settings.HEALTH_PROBE_PROJECT_ID
    if org_id is None or project_id is None:
        logger.warning(
            "[run_health_probe_tick] Health probe org/project not configured — skipping"
        )
        return {
            "enqueued": False,
            "job_id": None,
            "probe_index": None,
            "previous_job_status": None,
        }

    previous_status = _check_previous_probe(project_id)
    _capture_previous_result_checkin(previous_status)

    index = _claim_next_probe_index()
    probe = _PROBES[index]
    request = _build_probe_request(probe, index)

    with Session(engine) as session:
        job_id = start_job(
            db=session,
            request=request,
            project_id=project_id,
            organization_id=org_id,
        )

    _store_last_job_id(job_id)

    logger.info(
        f"[run_health_probe_tick] Fired probe | provider: {probe.provider}, "
        f"model: {probe.model}, modality: {probe.modality}, job_id: {job_id}, "
        f"previous_job_status: {previous_status}"
    )
    return {
        "enqueued": True,
        "job_id": job_id,
        "probe_index": index,
        "previous_job_status": previous_status.value if previous_status else None,
    }
