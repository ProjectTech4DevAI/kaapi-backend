import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sentry_sdk.crons import MonitorStatus, capture_checkin
from sentry_sdk.types import MonitorConfig
from sqlmodel import Session

from app.core.audio_utils import AudioRef
from app.core.config import settings
from app.core.db import engine
from app.models.llm import KaapiCompletionConfig, NativeCompletionConfig, QueryParams
from app.services.llm.mappers import transform_kaapi_config_to_native
from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.registry import get_llm_provider

logger = logging.getLogger(__name__)

_PROBE_INPUT = "ping"
_PROBE_MAX_TOKENS = 1
_PROBE_WORKERS = 4

# ~1sec "Hello" clip used as STT probe input.
_STT_AUDIO_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "health_probe_hello.ogg"
)
_STT_AUDIO_MIME = "audio/ogg"
_stt_audio_bytes: bytes | None = None

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


Modality = Literal["text", "tts", "stt"]


@dataclass(frozen=True)
class Probe:
    provider: str
    model: str
    modality: Modality


# (probe, provider, built config+input, build error) — exactly one of the last
# two is set; a build error short-circuits _run_probe before provider.execute.
BuiltProbe = tuple[NativeCompletionConfig, str | AudioRef]
PreparedProbe = tuple[Probe, BaseProvider | None, BuiltProbe | None, str | None]


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
    Probe(provider="sarvamai", model="bulbul:v3", modality="tts"),
    Probe(provider="elevenlabs", model="eleven_v3", modality="tts"),
    # STT
    Probe(provider="google", model="gemini-2.5-pro", modality="stt"),
    Probe(provider="google", model="gemini-2.5-flash", modality="stt"),
    Probe(provider="google", model="gemini-3.1-pro-preview", modality="stt"),
    Probe(provider="sarvamai", model="saaras:v3", modality="stt"),
    Probe(provider="elevenlabs", model="scribe_v2", modality="stt"),
]


def _load_stt_audio() -> bytes | None:
    global _stt_audio_bytes
    if _stt_audio_bytes is None:
        try:
            with open(_STT_AUDIO_PATH, "rb") as f:
                _stt_audio_bytes = f.read()
        except OSError as e:
            logger.warning(
                f"[_load_stt_audio] Probe audio unavailable | "
                f"path: {_STT_AUDIO_PATH}, error: {e}"
            )
            return None
    return _stt_audio_bytes


def _build_provider(
    session: Session, probe: Probe, *, org_id: int, project_id: int
) -> BaseProvider | None:
    try:
        return get_llm_provider(
            session=session,
            provider_type=probe.provider,
            project_id=project_id,
            organization_id=org_id,
        )
    except (ValueError, RuntimeError) as e:
        logger.warning(
            f"[_build_provider] Client init failed | provider: {probe.provider}, "
            f"modality: {probe.modality}, error: {e}"
        )
        return None


def _build_config_and_input(session: Session, probe: Probe) -> BuiltProbe | None:
    if probe.modality == "text":
        cfg = KaapiCompletionConfig.model_validate(
            {
                "provider": probe.provider,
                "type": "text",
                "params": {
                    "model": probe.model,
                    "temperature": 0.0,
                    "max_output_tokens": _PROBE_MAX_TOKENS,
                },
            }
        )
        resolved_input: str | AudioRef = _PROBE_INPUT
    elif probe.modality == "tts":
        cfg = KaapiCompletionConfig.model_validate(
            {
                "provider": probe.provider,
                "type": "tts",
                "params": {"model": probe.model},
            }
        )
        resolved_input = _PROBE_INPUT
    else:  # stt
        audio_bytes = _load_stt_audio()
        if audio_bytes is None:
            return None
        cfg = KaapiCompletionConfig.model_validate(
            {
                "provider": probe.provider,
                "type": "stt",
                "params": {"model": probe.model},
            }
        )
        resolved_input = AudioRef(bytes_=audio_bytes, mime_type=_STT_AUDIO_MIME)

    native_config, _ = transform_kaapi_config_to_native(
        session=session, kaapi_config=cfg
    )
    return native_config, resolved_input


def _run_probe(
    probe: Probe,
    provider: BaseProvider | None,
    built: BuiltProbe | None,
    build_error: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "endpoint": "llm/call",
        "provider": probe.provider,
        "modality": probe.modality,
        "model": probe.model,
        "ok": False,
        "latency_ms": None,
        "error": None,
    }
    if build_error is not None or provider is None or built is None:
        result["error"] = build_error or "not_prepared"
        return result

    config, resolved_input = built
    query = QueryParams(input=_PROBE_INPUT if isinstance(resolved_input, str) else "")

    started = time.perf_counter()
    try:
        response, error = provider.execute(
            completion_config=config,
            query=query,
            resolved_input=resolved_input,
        )
    except Exception as e:
        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error(
            f"[_run_probe] Raised | provider: {probe.provider}, "
            f"modality: {probe.modality}, error: {result['error']}"
        )
        return result

    result["latency_ms"] = int((time.perf_counter() - started) * 1000)
    if response is None:
        result["error"] = error or "no_response"
        logger.error(
            f"[_run_probe] Failed | provider: {probe.provider}, "
            f"modality: {probe.modality}, error: {result['error']}"
        )
        return result

    result["ok"] = True
    return result


def _prepare_probes(
    session: Session, *, org_id: int, project_id: int
) -> list[PreparedProbe]:
    provider_cache: dict[str, BaseProvider | None] = {}
    for p in _PROBES:
        if p.provider not in provider_cache:
            provider_cache[p.provider] = _build_provider(
                session, p, org_id=org_id, project_id=project_id
            )

    prepared: list[PreparedProbe] = []
    for p in _PROBES:
        provider = provider_cache[p.provider]
        if provider is None:
            prepared.append((p, None, None, "client_init_failed"))
            continue
        try:
            built = _build_config_and_input(session, p)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.error(
                f"[_prepare_probes] Config build failed | provider: {p.provider}, "
                f"modality: {p.modality}, error: {error}"
            )
            prepared.append((p, provider, None, error))
            continue
        if built is None:
            prepared.append((p, provider, None, "stt_audio_not_configured"))
            continue
        prepared.append((p, provider, built, None))
    return prepared


def run_probes() -> dict[str, Any]:
    org_id = settings.HEALTH_PROBE_ORG_ID
    project_id = settings.HEALTH_PROBE_PROJECT_ID
    if org_id is None or project_id is None:
        logger.warning(
            "[run_probes] Health probe org/project not configured — skipping"
        )
        return {"skipped": True, "reason": "health_probe_org_or_project_not_set"}

    logger.info(
        f"[run_probes] Starting | probes: {len(_PROBES)}, "
        f"org_id: {org_id}, project_id: {project_id}"
    )

    # Session covers only preparation (credential lookup + config transform);
    # it must be closed before the slow external provider calls below.
    with Session(engine) as session:
        prepared = _prepare_probes(session, org_id=org_id, project_id=project_id)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=_PROBE_WORKERS) as pool:
        results = list(pool.map(lambda pp: _run_probe(*pp), prepared))
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    ok_count = sum(1 for r in results if r["ok"])
    logger.info(
        f"[run_probes] Completed | total: {len(results)}, ok: {ok_count}, "
        f"failed: {len(results) - ok_count}, elapsed_ms: {elapsed_ms}"
    )
    return {
        "elapsed_ms": elapsed_ms,
        "total": len(results),
        "ok": ok_count,
        "failed": len(results) - ok_count,
        "results": results,
    }


def execute_health_probes() -> dict[str, Any]:
    """Run the probes wrapped in Sentry cron check-ins.

    The check-in happens here (not in the cron route) so the monitor reflects
    actual probe results rather than enqueue success.
    """
    check_in_id = capture_checkin(
        monitor_slug=HEALTH_PROBES_MONITOR_SLUG,
        status=MonitorStatus.IN_PROGRESS,
        monitor_config=HEALTH_PROBES_MONITOR_CONFIG,
    )
    try:
        result = run_probes()
    except Exception:
        capture_checkin(
            check_in_id=check_in_id,
            monitor_slug=HEALTH_PROBES_MONITOR_SLUG,
            status=MonitorStatus.ERROR,
        )
        raise

    ok = not result.get("skipped") and result.get("failed") == 0
    capture_checkin(
        check_in_id=check_in_id,
        monitor_slug=HEALTH_PROBES_MONITOR_SLUG,
        status=MonitorStatus.OK if ok else MonitorStatus.ERROR,
    )
    return result


def main() -> None:
    """Manual smoke test: `uv run python -m app.services.health_probes`.

    Calls run_probes directly (no Sentry check-ins) against the configured
    HEALTH_PROBE_ORG_ID/PROJECT_ID — real provider calls, real credentials.
    """
    import json

    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_probes(), indent=2))


if __name__ == "__main__":
    main()
