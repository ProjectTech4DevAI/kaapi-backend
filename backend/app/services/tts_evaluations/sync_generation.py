"""Celery task function for synchronous TTS synthesis (Sarvam, ElevenLabs).

One task claims every PENDING row of one model (moving them to PROCESSING) and
synthesizes them `settings.TTS_SYNC_MAX_WORKERS` rows at a time. Provider calls and
uploads run with no DB session open so a slow provider can't pin pooled connections. The task only writes result rows; the cron sets the run status.
"""

import base64
import io
import logging
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Required, TypedDict

import httpx
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError as ElevenLabsApiError
from gevent import Timeout
from sarvamai import SarvamAI
from sqlmodel import Session

from app.core.audio_utils import calculate_duration, pcm_to_wav
from app.core.cloud.storage import get_cloud_storage
from app.core.config import settings
from app.core.db import engine
from app.core.providers import Provider
from app.core.storage_utils import upload_to_object_store
from app.crud.credentials import get_provider_credential
from app.crud.tts_evaluations.result import (
    UNFINISHED_TTS_STATUSES,
    claim_pending_tts_results,
    fail_pending_tts_results,
    update_tts_result,
)
from app.models.job import JobStatus
from app.models.llm.constants import BCP47_TO_ELEVENLABS_LANG
from app.models.tts_evaluation import TTSUsageUnitEnum
from app.services.llm.mappers import voice_to_id
from app.services.llm.providers.eleven_ai import ElevenlabsAIProvider
from app.services.llm.providers.sarvam_ai import SarvamAIProvider
from app.services.tts_evaluations.constants import (
    DEFAULT_ELEVENLABS_VOICE,
    ELEVENLABS_TTS_MODELS,
    SARVAM_SPEAKER,
    SARVAM_TTS_MODELS,
    TTS_AUDIO_SUBDIRECTORY,
)
from app.services.tts_evaluations.voices import default_voice_for_model

logger = logging.getLogger(__name__)

# Every stored clip is 24 kHz mono 16-bit
TTS_SAMPLE_RATE = 24000


class TTSResultFields(TypedDict, total=False):
    """Keyword arguments for `update_tts_result` produced for one synthesized row."""

    status: Required[str]
    object_store_url: str
    metadata: dict[str, float | int]
    error_message: str
    usage: dict[str, Any]


def synthesize_sarvam(
    client: SarvamAI,
    text: str,
    model: str,
    language_code: str,
    *,
    voice: str = SARVAM_SPEAKER,
) -> bytes:
    """Synthesize text with Sarvam speaker `voice`; returns raw 24 kHz mono 16-bit PCM."""
    response = client.text_to_speech.convert(
        text=text,
        model=model,
        target_language_code=language_code,
        speaker=voice,
        speech_sample_rate=TTS_SAMPLE_RATE,
    )
    if not response.audios:
        raise ValueError("Sarvam returned no audio")

    # Long inputs come back as several base64 chunks, each a full WAV file; strip
    # every header so later RIFF headers don't end up in the PCM as clicks.
    pcm = b""
    for audio_b64 in response.audios:
        with wave.open(io.BytesIO(base64.b64decode(audio_b64)), "rb") as wav_file:
            pcm += wav_file.readframes(wav_file.getnframes())
    return pcm


def synthesize_elevenlabs(
    client: ElevenLabs,
    text: str,
    model: str,
    language_code: str,
    *,
    voice: str = DEFAULT_ELEVENLABS_VOICE,
) -> bytes:
    """Synthesize text with the ElevenLabs voice named `voice`; returns raw 24 kHz mono 16-bit PCM.

    Raises:
        ValueError: If `voice` has no known ElevenLabs voice ID.
    """
    voice_id = voice_to_id(voice)
    if not voice_id:
        raise ValueError(f"Unsupported ElevenLabs voice '{voice}'")

    chunks = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model,
        output_format=f"pcm_{TTS_SAMPLE_RATE}",
        # Unmapped languages (e.g. Odia) are left to ElevenLabs' auto-detect.
        language_code=BCP47_TO_ELEVENLABS_LANG.get(language_code),
    )
    pcm = b"".join(chunks)
    if not pcm:
        raise ValueError("ElevenLabs returned no audio")
    return pcm


def fetch_elevenlabs_credit_multiplier(client: ElevenLabs, model: str) -> float | None:
    """Return the model's character_cost_multiplier from the ElevenLabs models list, or None."""
    try:
        models = client.models.list()
    except (ElevenLabsApiError, httpx.HTTPError) as e:
        logger.warning(
            f"[fetch_elevenlabs_credit_multiplier] [ELEVENLABS] Models list failed; "
            f"credits will be null | model={model}, error={str(e)}"
        )
        return None

    for item in models:
        if item.model_id == model and item.model_rates is not None:
            return item.model_rates.character_cost_multiplier

    logger.warning(
        f"[fetch_elevenlabs_credit_multiplier] [ELEVENLABS] No rates for model; "
        f"credits will be null | model={model}"
    )
    return None


def execute_tts_sync_generation(
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance: Task,
    organization_id: int,
    model: str,
    language_code: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Claim and synthesize every PENDING Sarvam/ElevenLabs result of one model.

    Rows are claimed (PENDING -> PROCESSING) before any work, so a duplicate or
    redelivered task claims nothing and exits.

    Args:
        project_id: Project ID
        job_id: Evaluation run ID (as string)
        task_id: Celery task ID
        task_instance: Celery task instance
        organization_id: Organization ID
        model: Sarvam or ElevenLabs model
        language_code: BCP-47 tag resolved from the dataset (e.g. hi-IN)

    Returns:
        dict: Summary with processed/failed counts
    """
    run_id = int(job_id)
    if model in SARVAM_TTS_MODELS:
        provider = Provider.SARVAMAI
    elif model in ELEVENLABS_TTS_MODELS:
        provider = Provider.ELEVENLABS
    else:
        error_message = f"Unsupported sync TTS model: {model}"
        logger.error(f"[execute_tts_sync_generation] {error_message} | run_id={run_id}")
        with Session(engine) as session:
            fail_pending_tts_results(
                session=session,
                run_id=run_id,
                provider=model,
                error_message=error_message,
            )
        return {"success": False, "run_id": run_id, "error": error_message}

    logger.info(
        f"[execute_tts_sync_generation] Starting | run_id={run_id}, model={model}, "
        f"language={language_code}, celery_task_id={task_id}"
    )

    try:
        with Session(engine) as session:
            claimed = claim_pending_tts_results(
                session=session, run_id=run_id, provider=model
            )
            if not claimed:
                logger.info(
                    f"[execute_tts_sync_generation] No pending rows to claim | "
                    f"run_id={run_id}, model={model}"
                )
                return {"success": True, "run_id": run_id, "processed": 0, "failed": 0}

            pending_rows: list[tuple[int, str, str]] = []
            for result in claimed:
                # NULL voice: row predates per-model voices and used the default.
                voice = result.voice or default_voice_for_model(model)
                pending_rows.append((result.id, result.sample_text, voice))

            credentials = get_provider_credential(
                session=session,
                org_id=organization_id,
                project_id=project_id,
                provider=provider.value,
            )
            storage = get_cloud_storage(session=session, project_id=project_id)

        if not isinstance(credentials, dict):
            raise ValueError(f"{provider.value} credentials are not configured")

        credit_multiplier: float | None = None
        if model in SARVAM_TTS_MODELS:
            client = SarvamAIProvider.create_client(credentials)
        else:
            client = ElevenlabsAIProvider.create_client(credentials)
            credit_multiplier = fetch_elevenlabs_credit_multiplier(client, model)

        def synthesize_and_upload(
            result_id: int, text: str, voice: str
        ) -> TTSResultFields:
            """Return `update_tts_result` fields; never raises, so one row can't stop the rest."""
            try:
                if model in SARVAM_TTS_MODELS:
                    pcm = synthesize_sarvam(
                        client, text, model, language_code, voice=voice
                    )
                else:
                    pcm = synthesize_elevenlabs(
                        client, text, model, language_code, voice=voice
                    )
                wav = pcm_to_wav(pcm)
                audio_url = upload_to_object_store(
                    storage=storage,
                    content=wav,
                    filename=f"{uuid.uuid4()}.wav",
                    subdirectory=TTS_AUDIO_SUBDIRECTORY,
                    content_type="audio/wav",
                )
            except Exception as e:
                logger.warning(
                    f"[execute_tts_sync_generation] Synthesis failed | "
                    f"run_id={run_id}, model={model}, result_id={result_id}, "
                    f"voice={voice}, error={str(e)}"
                )
                return {
                    "status": JobStatus.FAILED.value,
                    "error_message": f"{provider.value} synthesis failed: {str(e)}",
                }

            if not audio_url:
                return {
                    "status": JobStatus.FAILED.value,
                    "error_message": "Audio upload to object store failed",
                }

            # Neither API reports per-request usage: both bill per input character,
            # and ElevenLabs credits are characters x the model's cost multiplier.
            characters = len(text)
            usage: dict[str, Any]
            if model in SARVAM_TTS_MODELS:
                usage = {
                    "unit": TTSUsageUnitEnum.CHARACTERS.value,
                    "characters": characters,
                }
            else:
                credits = None
                if credit_multiplier is not None:
                    credits = characters * credit_multiplier
                usage = {
                    "unit": TTSUsageUnitEnum.CREDITS.value,
                    "characters": characters,
                    "credits": credits,
                    "credit_multiplier": credit_multiplier,
                }
            return {
                "status": JobStatus.SUCCESS.value,
                "object_store_url": audio_url,
                "metadata": {
                    "duration_seconds": round(calculate_duration(len(pcm)), 3),
                    "size_bytes": len(wav),
                },
                "usage": usage,
            }

        processed = 0
        failed = 0
        # Shut down by hand: a `with` block would wait for in-flight provider calls on timeout.
        executor = ThreadPoolExecutor(max_workers=settings.TTS_SYNC_MAX_WORKERS)
        try:
            futures: dict[Future[TTSResultFields], int] = {}
            for result_id, text, voice in pending_rows:
                future = executor.submit(synthesize_and_upload, result_id, text, voice)
                futures[future] = result_id

            for future in as_completed(futures):
                fields = future.result()
                with Session(engine) as session:
                    updated = update_tts_result(
                        session=session, result_id=futures[future], **fields
                    )
                    session.commit()

                if updated is None:
                    logger.warning(
                        f"[execute_tts_sync_generation] Result no longer processing, "
                        f"write skipped | run_id={run_id}, model={model}, "
                        f"result_id={futures[future]}"
                    )
                elif fields["status"] == JobStatus.SUCCESS.value:
                    processed += 1
                else:
                    failed += 1
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        logger.info(
            f"[execute_tts_sync_generation] Completed | run_id={run_id}, model={model}, "
            f"processed={processed}, failed={failed}"
        )
        return {
            "success": True,
            "run_id": run_id,
            "processed": processed,
            "failed": failed,
        }

    except (Timeout, SoftTimeLimitExceeded):
        logger.warning(
            f"[execute_tts_sync_generation] Timed out | run_id={run_id}, model={model}"
        )
        with Session(engine) as session:
            fail_pending_tts_results(
                session=session,
                run_id=run_id,
                provider=model,
                error_message="Synthesis timed out before this sample was processed",
                statuses=UNFINISHED_TTS_STATUSES,
            )
        raise

    except Exception as e:
        logger.error(
            f"[execute_tts_sync_generation] Failed | run_id={run_id}, model={model}, "
            f"error={str(e)}",
            exc_info=True,
        )
        with Session(engine) as session:
            fail_pending_tts_results(
                session=session,
                run_id=run_id,
                provider=model,
                error_message=f"{provider.value} synthesis failed: {str(e)}",
                statuses=UNFINISHED_TTS_STATUSES,
            )
        return {"success": False, "run_id": run_id, "error": str(e)}
