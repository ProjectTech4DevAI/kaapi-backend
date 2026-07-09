import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from sqlmodel import Session

from app.core.audio_utils import AudioRef
from app.core.config import settings
from app.models.llm import KaapiCompletionConfig, QueryParams
from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.registry import get_llm_provider

logger = logging.getLogger(__name__)

_PROBE_INPUT = "ping"
_PROBE_MAX_TOKENS = 1
_PROBE_WORKERS = 4

# "Hello" ~1sec
_STT_AUDIO_B64: str = "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"
_STT_AUDIO_MIME: str = "audio/ogg"


Modality = Literal["text", "tts", "stt"]


@dataclass(frozen=True)
class Probe:
    provider: str
    model: str
    modality: Modality


_PROBES: list[Probe] = [
    # Text
    Probe(provider="openai", model="gpt-4o-1-mini", modality="text"),
    Probe(provider="google", model="gemini-2.5-flash", modality="text"),
    Probe(provider="google-aistudio", model="gemini-2.5-flash", modality="text"),
    Probe(provider="anthropic", model="claude-3-5-haiku-latest", modality="text"),
    Probe(provider="sarvamai", model="sarvam-m", modality="text"),
    # TTS
    Probe(provider="openai", model="tts-1", modality="tts"),
    Probe(provider="google", model="gemini-2.5-flash-preview-ttsss", modality="tts"),
    Probe(provider="sarvamai", model="bulbul:v2", modality="tts"),
    Probe(provider="elevenlabs", model="eleven_flash_v2_5", modality="tts"),
    # STT
    Probe(provider="openai", model="whisper-1", modality="stt"),
    Probe(provider="google", model="gemini-2.5-pro", modality="stt"),
    Probe(provider="sarvamai", model="saarika:v2.5", modality="stt"),
    Probe(provider="elevenlabs", model="scribe_v1", modality="stt"),
]


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
        logger.error(
            f"[_build_provider] Client init failed | provider: {probe.provider}, "
            f"modality: {probe.modality}, error: {e}"
        )
        return None


def _build_config_and_input(
    probe: Probe,
) -> tuple[KaapiCompletionConfig, str | AudioRef] | None:
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
        return cfg, _PROBE_INPUT

    if probe.modality == "tts":
        cfg = KaapiCompletionConfig.model_validate(
            {
                "provider": probe.provider,
                "type": "tts",
                "params": {"model": probe.model},
            }
        )
        return cfg, _PROBE_INPUT

    # stt
    if not _STT_AUDIO_B64:
        return None
    cfg = KaapiCompletionConfig.model_validate(
        {
            "provider": probe.provider,
            "type": "stt",
            "params": {"model": probe.model},
        }
    )
    audio = AudioRef(
        bytes_=base64.b64decode(_STT_AUDIO_B64),
        mime_type=_STT_AUDIO_MIME,
    )
    return cfg, audio


def _run_probe(provider: BaseProvider | None, probe: Probe) -> dict[str, Any]:
    result: dict[str, Any] = {
        "endpoint": "llm/call",
        "provider": probe.provider,
        "modality": probe.modality,
        "model": probe.model,
        "ok": False,
        "latency_ms": None,
        "error": None,
    }
    if provider is None:
        result["error"] = "client_init_failed"
        return result

    built = _build_config_and_input(probe)
    if built is None:
        result["error"] = "stt_audio_not_configured"
        return result
    config, resolved_input = built
    query = QueryParams(input=_PROBE_INPUT if isinstance(resolved_input, str) else "")

    started = time.perf_counter()
    try:
        response, error = provider.execute(
            completion_config=config,  # type: ignore[arg-type]
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


def run_probes(*, session: Session) -> dict[str, Any]:
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

    provider_cache: dict[str, BaseProvider | None] = {}
    for p in _PROBES:
        if p.provider not in provider_cache:
            provider_cache[p.provider] = _build_provider(
                session, p, org_id=org_id, project_id=project_id
            )
    dispatch = [(p, provider_cache[p.provider]) for p in _PROBES]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=_PROBE_WORKERS) as pool:
        results = list(pool.map(lambda pp: _run_probe(pp[1], pp[0]), dispatch))
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


if __name__ == "__main__":
    assert _PROBES, "probe list must not be empty"
    modalities = {p.modality for p in _PROBES}
    assert modalities == {"text", "tts", "stt"}, modalities
    for probe in _PROBES:
        if probe.modality == "stt" and not _STT_AUDIO_B64:
            continue
        built = _build_config_and_input(probe)
        assert built is not None
    print("ok")
