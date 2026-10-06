"""Celery task function for synchronous (non-batch) TTS evaluation synthesis.

One task handles one model and a slice of that model's result IDs. The DB session
is only held for short reads/writes; provider calls and S3 uploads run with no
session so a slow provider can't pin pooled connections.
"""

import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session

from app.core.cloud.storage import CloudStorage, get_cloud_storage
from app.core.db import engine
from app.core.providers import Provider
from app.core.storage_utils import upload_to_object_store
from app.crud.credentials import get_provider_credential
from app.crud.tts_evaluations.result import (
    bulk_update_pending_tts_results,
    list_pending_tts_results_by_ids,
)
from app.crud.tts_evaluations.run import finalize_tts_run_status
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResultUpdate
from app.services.tts_evaluations.constants import (
    KAAPI_TAG,
    TTS_AUDIO_CONTENT_TYPE,
    TTS_AUDIO_FILE_EXTENSION,
    TTS_AUDIO_SUBDIRECTORY,
    TTS_SYNC_MAX_WORKERS,
    get_tts_model_spec,
)
from app.services.tts_evaluations.synthesizers import (
    TTSClient,
    TTSSynthesisError,
    create_tts_client,
    synthesize_tts,
)

logger = logging.getLogger(__name__)

DURATION_DECIMALS = 3


@dataclass(frozen=True)
class _PendingItem:
    result_id: int
    text: str


@dataclass(frozen=True)
class _ChunkContext:
    items: list[_PendingItem]
    provider: Provider
    voice: str
    client: TTSClient | None
    storage: CloudStorage | None
    setup_error: str | None


def execute_tts_sync_chunk(
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance: Task,
    organization_id: int,
    model: str,
    result_ids: list[int],
    language_code: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Synthesize one chunk of sync-model TTS results in a Celery worker.

    Args:
        project_id: Project ID
        job_id: Evaluation run ID (as string)
        task_id: Celery task ID
        task_instance: Celery task instance
        organization_id: Organization ID
        model: Sync-mode TTS model (e.g. bulbul:v3, eleven_v3)
        result_ids: TTSResult IDs in this chunk (all for `model`)
        language_code: BCP-47 tag resolved from the dataset (e.g. hi-IN)

    Returns:
        dict: Summary with processed/failed counts and the run status
    """
    run_id = int(job_id)

    logger.info(
        f"[execute_tts_sync_chunk] Starting | run_id={run_id}, model={model}, "
        f"chunk_size={len(result_ids)}, language={language_code}, "
        f"celery_task_id={task_id}"
    )

    try:
        context = _prepare_chunk(
            run_id=run_id,
            organization_id=organization_id,
            project_id=project_id,
            model=model,
            result_ids=result_ids,
        )

        if not context.items:
            logger.info(
                f"[execute_tts_sync_chunk] No pending results, skipping | "
                f"run_id={run_id}, model={model}"
            )
            run_status = _finalize_run(run_id)
            return {
                "success": True,
                "run_id": run_id,
                "processed": 0,
                "failed": 0,
                "run_status": run_status,
            }

        if context.setup_error is not None:
            failed_count = _fail_pending_results(
                run_id=run_id,
                result_ids=result_ids,
                error_message=context.setup_error,
            )
            run_status = _finalize_run(run_id)
            return {
                "success": False,
                "run_id": run_id,
                "processed": 0,
                "failed": failed_count,
                "run_status": run_status,
                "error": context.setup_error,
            }

        processed_count, failed_count = _synthesize_chunk(
            context=context,
            run_id=run_id,
            model=model,
            language_code=language_code,
        )

        run_status = _finalize_run(run_id)

        logger.info(
            f"[execute_tts_sync_chunk] Completed | run_id={run_id}, model={model}, "
            f"processed={processed_count}, failed={failed_count}, "
            f"run_status={run_status}"
        )

        return {
            "success": True,
            "run_id": run_id,
            "processed": processed_count,
            "failed": failed_count,
            "run_status": run_status,
        }

    except (Timeout, SoftTimeLimitExceeded):
        logger.warning(
            f"[execute_tts_sync_chunk] Chunk timed out | run_id={run_id}, model={model}"
        )
        # Nothing re-drives a sync chunk, so leftover PENDING rows would strand the run.
        _fail_pending_results(
            run_id=run_id,
            result_ids=result_ids,
            error_message=(
                f"[{KAAPI_TAG}] Synthesis timed out before this sample was processed. "
                f"Re-run the evaluation; if it persists, contact Kaapi."
            ),
        )
        _finalize_run(run_id)
        raise

    except Exception as e:
        logger.error(
            f"[execute_tts_sync_chunk] Failed | run_id={run_id}, model={model}, "
            f"error={str(e)}",
            exc_info=True,
        )
        _fail_pending_results(
            run_id=run_id,
            result_ids=result_ids,
            error_message=(
                f"[{KAAPI_TAG}] Synthesis chunk failed unexpectedly: {str(e)}. "
                f"Contact Kaapi if the issue persists."
            ),
        )
        run_status = _finalize_run(run_id)
        return {
            "success": False,
            "run_id": run_id,
            "error": str(e),
            "run_status": run_status,
        }


def _prepare_chunk(
    *,
    run_id: int,
    organization_id: int,
    project_id: int,
    model: str,
    result_ids: list[int],
) -> _ChunkContext:
    """Load pending items, credentials, client and storage in one short session."""
    spec = get_tts_model_spec(model)

    with Session(engine) as session:
        pending_rows = list_pending_tts_results_by_ids(
            session=session, run_id=run_id, result_ids=result_ids
        )
        items: list[_PendingItem] = []
        for row in pending_rows:
            items.append(_PendingItem(result_id=row.id, text=row.sample_text))

        if not items:
            return _ChunkContext(
                items=items,
                provider=spec.provider,
                voice=spec.default_voice,
                client=None,
                storage=None,
                setup_error=None,
            )

        credentials = get_provider_credential(
            session=session,
            org_id=organization_id,
            project_id=project_id,
            provider=spec.provider.value,
        )
        storage = get_cloud_storage(session=session, project_id=project_id)

    if not isinstance(credentials, dict):
        setup_error = (
            f"[{KAAPI_TAG}] {spec.provider.value} credentials are not configured "
            f"for this project. Add them under credentials and re-run the evaluation."
        )
        logger.warning(
            f"[_prepare_chunk] {setup_error} | run_id={run_id}, model={model}"
        )
        return _ChunkContext(
            items=items,
            provider=spec.provider,
            voice=spec.default_voice,
            client=None,
            storage=storage,
            setup_error=setup_error,
        )

    try:
        client = create_tts_client(spec.provider, credentials)
    except ValueError as e:
        setup_error = (
            f"[{KAAPI_TAG}] Invalid {spec.provider.value} credentials: {str(e)}. "
            f"Verify the stored API key and re-run the evaluation."
        )
        logger.warning(
            f"[_prepare_chunk] {setup_error} | run_id={run_id}, model={model}",
            exc_info=True,
        )
        return _ChunkContext(
            items=items,
            provider=spec.provider,
            voice=spec.default_voice,
            client=None,
            storage=storage,
            setup_error=setup_error,
        )

    return _ChunkContext(
        items=items,
        provider=spec.provider,
        voice=spec.default_voice,
        client=client,
        storage=storage,
        setup_error=None,
    )


def _synthesize_chunk(
    *,
    context: _ChunkContext,
    run_id: int,
    model: str,
    language_code: str,
) -> tuple[int, int]:
    """Fan synthesis out over a bounded pool; persist each outcome as it lands.

    Returns:
        tuple[int, int]: (processed_count, failed_count) actually written
    """
    if context.client is None or context.storage is None:
        raise ValueError("Chunk context is missing its client or storage")

    client = context.client
    storage = context.storage
    processed_count = 0
    failed_count = 0
    workers = max(1, min(TTS_SYNC_MAX_WORKERS, len(context.items)))

    # Managed by hand rather than `with`: on a gevent timeout, `with` would block
    # in shutdown(wait=True) until every in-flight provider call returned.
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures: list[Future[TTSResultUpdate]] = []
        for item in context.items:
            future = executor.submit(
                _synthesize_and_upload,
                client=client,
                storage=storage,
                provider=context.provider,
                model=model,
                voice=context.voice,
                language_code=language_code,
                item=item,
            )
            futures.append(future)

        for future in as_completed(futures):
            update = future.result()
            written = _write_result_updates([update])
            if not written:
                continue
            if update.status == JobStatus.SUCCESS:
                processed_count += 1
            else:
                failed_count += 1
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    logger.info(
        f"[_synthesize_chunk] Chunk synthesized | run_id={run_id}, model={model}, "
        f"processed={processed_count}, failed={failed_count}"
    )
    return processed_count, failed_count


def _synthesize_and_upload(
    *,
    client: TTSClient,
    storage: CloudStorage,
    provider: Provider,
    model: str,
    voice: str,
    language_code: str,
    item: _PendingItem,
) -> TTSResultUpdate:
    """Synthesize + upload one item. Never raises: failures become a FAILED update."""
    try:
        audio = synthesize_tts(
            client=client,
            provider=provider,
            text=item.text,
            model=model,
            language_code=language_code,
            voice=voice,
        )
    except TTSSynthesisError as e:
        return TTSResultUpdate(
            result_id=item.result_id,
            status=JobStatus.FAILED,
            error_message=e.message,
        )
    except Exception as e:
        # A worker exception would abort the whole chunk via future.result().
        error_message = (
            f"[{KAAPI_TAG}] Unexpected error during {provider.value} synthesis: "
            f"{str(e)}. Contact Kaapi if the issue persists."
        )
        logger.error(
            f"[_synthesize_and_upload] {error_message} | "
            f"result_id={item.result_id}, model={model}",
            exc_info=True,
        )
        return TTSResultUpdate(
            result_id=item.result_id,
            status=JobStatus.FAILED,
            error_message=error_message,
        )

    audio_url = upload_to_object_store(
        storage=storage,
        content=audio.wav_bytes,
        filename=f"{uuid.uuid4()}.{TTS_AUDIO_FILE_EXTENSION}",
        subdirectory=TTS_AUDIO_SUBDIRECTORY,
        content_type=TTS_AUDIO_CONTENT_TYPE,
    )

    if not audio_url:
        error_message = (
            f"[{KAAPI_TAG}] Audio upload to object store failed. Check the "
            f"project's storage configuration; contact Kaapi if it persists."
        )
        logger.error(
            f"[_synthesize_and_upload] {error_message} | "
            f"result_id={item.result_id}, model={model}"
        )
        return TTSResultUpdate(
            result_id=item.result_id,
            status=JobStatus.FAILED,
            error_message=error_message,
        )

    return TTSResultUpdate(
        result_id=item.result_id,
        status=JobStatus.SUCCESS,
        object_store_url=audio_url,
        metadata={
            "duration_seconds": round(audio.duration_seconds, DURATION_DECIMALS),
            "size_bytes": len(audio.wav_bytes),
        },
    )


def _write_result_updates(updates: list[TTSResultUpdate]) -> int:
    with Session(engine) as session:
        return bulk_update_pending_tts_results(session=session, updates=updates)


def _fail_pending_results(
    *, run_id: int, result_ids: list[int], error_message: str
) -> int:
    """Mark every still-PENDING result of this chunk FAILED in a fresh session."""
    with Session(engine) as session:
        pending_rows = list_pending_tts_results_by_ids(
            session=session, run_id=run_id, result_ids=result_ids
        )
        updates: list[TTSResultUpdate] = []
        for row in pending_rows:
            updates.append(
                TTSResultUpdate(
                    result_id=row.id,
                    status=JobStatus.FAILED,
                    error_message=error_message,
                )
            )
        return bulk_update_pending_tts_results(session=session, updates=updates)


def _finalize_run(run_id: int) -> str | None:
    with Session(engine) as session:
        run = finalize_tts_run_status(session=session, run_id=run_id)
        return run.status if run else None
