import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import httpx
import redis
import sentry_sdk
from sentry_sdk.crons import MonitorStatus, capture_checkin
from sentry_sdk.types import MonitorConfig
from sqlmodel import Session

from app.core.config import settings
from app.core.db import engine
from app.models.job import Job, JobStatus

logger = logging.getLogger(__name__)

PROBE_INPUT = "ping"

# ~1sec "Hello" clip used as the STT probe's input audio.
STT_AUDIO_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "health_probe_hello.ogg"
)
STT_AUDIO_MIME = "audio/ogg"
stt_audio_cache: dict[str, str | None] = {"value": None}

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

# Rotation state lives in Redis
LAST_JOB_ID_KEY = "health_probe:last_job_id"
INDEX_KEY = "health_probe:index"

redis_client: redis.Redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

Modality = Literal["text", "tts", "stt"]


@dataclass(frozen=True)
class Probe:
    provider: str
    model: str
    modality: Modality
    # Extra Kaapi params a provider's mapper requires beyond `model`
    # (Sarvam TTS needs `language`, ElevenLabs TTS needs `voice`).
    params: dict[str, Any] = field(default_factory=dict)


PROBES: list[Probe] = [
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
        params={"voice": "simran", "language": "en-IN"},
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


def load_stt_audio_b64() -> str:
    if stt_audio_cache["value"] is None:
        with open(STT_AUDIO_PATH, "rb") as f:
            stt_audio_cache["value"] = base64.b64encode(f.read()).decode("ascii")
    return stt_audio_cache["value"]


def build_probe_payload(probe: Probe, index: int) -> dict[str, Any]:
    if probe.modality == "stt":
        query_input: Any = {
            "type": "audio",
            "content": {
                "format": "base64",
                "value": load_stt_audio_b64(),
                "mime_type": STT_AUDIO_MIME,
            },
        }
    else:
        query_input = PROBE_INPUT

    # No callback_url: the probe reads its result by polling `Job.status` on
    # the next tick, so callback delivery (a best-effort side channel that
    # never affects Job.status) buys nothing here.
    return {
        "query": {"input": query_input},
        "config": {
            "blob": {
                "completion": {
                    "provider": probe.provider,
                    "type": probe.modality,
                    "params": {"model": probe.model, **probe.params},
                }
            }
        },
        "include_provider_raw_response": False,
        "request_metadata": {
            "health_probe": True,
            "probe_index": index,
            "provider": probe.provider,
            "modality": probe.modality,
            "model": probe.model,
        },
    }


def call_llm_probe(payload: dict[str, Any]) -> UUID:
    if not settings.HEALTH_PROBE_API_KEY or not settings.HEALTH_PROBE_LLM_CALL_URL:
        raise RuntimeError(
            "[call_llm_probe] HEALTH_PROBE_API_KEY and HEALTH_PROBE_LLM_CALL_URL "
            "must both be set to fire a health probe"
        )

    headers = {"X-API-KEY": settings.HEALTH_PROBE_API_KEY}
    response = httpx.post(
        settings.HEALTH_PROBE_LLM_CALL_URL,
        json=payload,
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    body = response.json()
    job_id_str = body["data"]["job_id"]
    return UUID(job_id_str)


def check_previous_probe() -> JobStatus | None:
    # Missing key (first run, Redis eviction) means skip the check-in, not fail the tick.
    try:
        last_job_id = redis_client.get(LAST_JOB_ID_KEY)
    except redis.RedisError as e:
        logger.warning(f"[check_previous_probe] Redis GET failed | error: {e}")
        sentry_sdk.capture_exception(e, level="warning")
        return None

    if last_job_id is None:
        logger.warning(
            f"[check_previous_probe] {LAST_JOB_ID_KEY} missing — skipping check-in"
        )
        sentry_sdk.capture_message(
            f"[check_previous_probe] {LAST_JOB_ID_KEY} missing — skipping check-in",
            level="warning",
        )
        return None

    try:
        job_uuid = UUID(str(last_job_id))
    except ValueError as e:
        logger.warning(
            f"[check_previous_probe] Malformed job id | value: {last_job_id!r}, error: {e}"
        )
        sentry_sdk.capture_exception(e, level="warning")
        return None

    with Session(engine) as session:
        job = session.get(Job, job_uuid)

    if job is None:
        logger.warning(f"[check_previous_probe] Job not found | job_id: {last_job_id}")
        sentry_sdk.capture_message(
            f"[check_previous_probe] Job not found | job_id: {last_job_id}",
            level="warning",
        )
        return None

    return job.status


def capture_previous_result_checkin(previous_status: JobStatus | None) -> None:
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


def claim_next_probe_index() -> int:
    # Redis INCR is atomic, so two overlapping ticks can never claim the same
    # slot (which would fire the same probe twice while skipping another).
    try:
        count = redis_client.incr(INDEX_KEY)
    except redis.RedisError as e:
        logger.warning(f"[claim_next_probe_index] Redis INCR failed | error: {e}")
        sentry_sdk.capture_exception(e, level="warning")
        return 0

    try:
        return (int(count) - 1) % len(PROBES)
    except (TypeError, ValueError) as e:
        logger.warning(f"[claim_next_probe_index] Non-integer index | raw: {count!r}")
        sentry_sdk.capture_exception(e, level="warning")
        return 0


def store_last_job_id(job_id: UUID) -> None:
    try:
        redis_client.set(LAST_JOB_ID_KEY, str(job_id))
    except redis.RedisError as e:
        logger.error(
            f"[store_last_job_id] Redis SET failed | job_id: {job_id}, error: {e}"
        )
        sentry_sdk.capture_exception(e)


def run_health_probe_tick() -> dict[str, Any]:
    previous_status = check_previous_probe()
    capture_previous_result_checkin(previous_status)

    index = claim_next_probe_index()
    probe = PROBES[index]
    payload = build_probe_payload(probe, index)

    job_id = call_llm_probe(payload)

    store_last_job_id(job_id)

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
