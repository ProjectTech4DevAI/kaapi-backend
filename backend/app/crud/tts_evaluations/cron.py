"""Cron processing functions for TTS evaluations.

The cron is the only code that sets a TTS run's final status. Each tick it polls
Gemini batches (dispatching result processing on success), fails the rows of
batches that ended badly or of sync models whose worker is gone, then recomputes
the run status from its result rows via `finalize_tts_run_status`.
"""

import logging
from datetime import timedelta
from typing import Any

from sqlmodel import Session, select

from app.celery.utils import start_tts_result_processing
from app.core.batch import BatchJobState, GeminiBatchProvider, GeminiClient
from app.core.util import now
from app.crud.evaluations.cron_utils import (
    TERMINAL_STATES,
    get_batch_jobs_for_run,
    make_empty_summary,
    make_poll_result,
    poll_batch_jobs,
)
from app.crud.tts_evaluations.result import (
    get_pending_results_for_run,
    update_tts_result,
)
from app.crud.tts_evaluations.run import finalize_tts_run_status
from app.models import EvaluationRun
from app.models.batch_job import BatchJob, BatchJobType
from app.models.job import JobStatus
from app.models.stt_evaluation import EvaluationType
from app.services.tts_evaluations.constants import (
    GEMINI_TTS_MODELS,
    SYNC_TTS_MODELS,
    TTS_SYNC_STALE_AFTER_SECONDS,
)

logger = logging.getLogger(__name__)

EVAL_TYPE_LABEL = "tts"
SUCCESS_ACTIONS = ("completed", "dispatched")


async def poll_all_pending_tts_evaluations(
    session: Session,
) -> dict[str, Any]:
    """Poll every processing TTS run, including runs with only Sarvam/ElevenLabs models.

    Not built on `poll_all_pending_evaluations_by_type`: that helper skips runs
    without a `batch_job_id` and fails a whole project when its Gemini client
    can't be built, which would break runs that don't use Gemini at all.

    Args:
        session: Database session

    Returns:
        Summary dict with total, processed, failed, still_processing counts
    """
    logger.info("[poll_all_pending_tts_evaluations] Starting TTS evaluation polling")

    statement = select(EvaluationRun).where(
        EvaluationRun.type == EvaluationType.TTS.value,
        EvaluationRun.status == "processing",
    )
    pending_runs = list(session.exec(statement).all())

    if not pending_runs:
        logger.info("[poll_all_pending_tts_evaluations] No pending TTS runs found")
        return make_empty_summary()

    all_results: list[dict[str, Any]] = []
    total_processed = 0
    total_failed = 0
    total_still_processing = 0

    for run in pending_runs:
        try:
            result = await poll_tts_run(
                session=session, run=run, org_id=run.organization_id
            )
        except Exception as e:
            # Left in `processing` so the next tick retries; a transient poll error
            # must not fail models that may still succeed.
            logger.error(
                f"[poll_all_pending_tts_evaluations] Failed to poll TTS run | "
                f"run_id={run.id} | {e}",
                exc_info=True,
            )
            session.rollback()
            result = make_poll_result(
                run=run,
                eval_type=EVAL_TYPE_LABEL,
                previous_status=run.status,
                current_status=run.status,
                action="no_change",
                error=f"Polling failed: {str(e)}",
            )
        all_results.append(result)

        if result["action"] in SUCCESS_ACTIONS:
            total_processed += 1
        elif result["action"] == "failed":
            total_failed += 1
        else:
            total_still_processing += 1

    logger.info(
        f"[poll_all_pending_tts_evaluations] Polling summary | "
        f"processed={total_processed} | failed={total_failed} | "
        f"still_processing={total_still_processing}"
    )

    return {
        "total": len(pending_runs),
        "processed": total_processed,
        "failed": total_failed,
        "still_processing": total_still_processing,
        "details": all_results,
    }


def _dispatch_tts_result_processing(
    run: EvaluationRun,
    batch_job: BatchJob,
    org_id: int,
    provider_name: str,
) -> str:
    """Dispatch TTS result processing to Celery low priority queue.

    Args:
        run: The evaluation run
        batch_job: The batch job record
        org_id: Organization ID
        provider_name: TTS provider/model name

    Returns:
        str: Celery task ID
    """
    celery_task_id = start_tts_result_processing(
        project_id=run.project_id,
        job_id=str(batch_job.id),
        organization_id=org_id,
        evaluation_run_id=run.id,
        tts_provider=provider_name,
        provider_batch_id=batch_job.provider_batch_id,
    )

    logger.info(
        f"[_dispatch_tts_result_processing] Dispatched to Celery | "
        f"run_id={run.id}, batch_job_id={batch_job.id}, "
        f"provider={provider_name}, celery_task_id={celery_task_id}"
    )

    return celery_task_id


async def poll_tts_run(
    session: Session,
    run: EvaluationRun,
    org_id: int,
) -> dict[str, Any]:
    """Advance a single TTS run and recompute its status.

    Args:
        session: Database session
        run: The evaluation run to poll
        org_id: Organization ID

    Returns:
        dict: Status result with run details and action taken
    """
    log_prefix = (
        f"[poll_tts_run] [org={org_id}][project={run.project_id}][eval={run.id}]"
    )
    logger.info(f"{log_prefix} Polling run")

    previous_status = run.status
    models = run.providers or []
    batch_models = [model for model in models if model in GEMINI_TTS_MODELS]
    sync_models = [model for model in models if model in SYNC_TTS_MODELS]
    # Both timestamps are naive UTC (`now()` strips tzinfo before storing).
    is_stale = now() - run.inserted_at > timedelta(seconds=TTS_SYNC_STALE_AFTER_SECONDS)
    any_dispatched = False

    batch_jobs: list[BatchJob] = []
    if batch_models:
        # A job whose creation failed has no provider_batch_id and nothing to poll;
        # submission already marked its rows FAILED.
        batch_jobs = [
            batch_job
            for batch_job in get_batch_jobs_for_run(
                session=session, run=run, job_type=BatchJobType.TTS_EVALUATION
            )
            if batch_job.provider_batch_id
        ]

    # Submission marks rows FAILED when a batch can't be created, so leftover PENDING
    # rows without a batch job mean the submission task died part-way.
    if batch_models and not batch_jobs and is_stale:
        logger.warning(f"{log_prefix} No batch jobs found for Gemini models")
        for model in batch_models:
            _fail_pending_results(
                session=session,
                run_id=run.id,
                model=model,
                error_message="No batch jobs found",
            )

    if batch_jobs:
        batch_provider: GeminiBatchProvider | None = None
        try:
            gemini_client = GeminiClient.from_credentials(
                session=session, org_id=org_id, project_id=run.project_id
            )
            batch_provider = GeminiBatchProvider(client=gemini_client.client)
        except Exception as client_err:
            logger.error(
                f"{log_prefix} Failed to get Gemini client | error={client_err}"
            )
            for model in batch_models:
                _fail_pending_results(
                    session=session,
                    run_id=run.id,
                    model=model,
                    error_message=f"Gemini client initialization failed: {str(client_err)}",
                )

        if batch_provider is not None:

            async def _on_batch_succeeded(
                batch_job: BatchJob, provider_name: str
            ) -> bool:
                _dispatch_tts_result_processing(run, batch_job, org_id, provider_name)
                return True

            async def _on_already_succeeded(
                batch_job: BatchJob, provider_name: str
            ) -> bool:
                pending = get_pending_results_for_run(
                    session=session, run_id=run.id, provider=provider_name
                )
                if not pending:
                    return False
                logger.info(
                    f"{log_prefix} Dispatching reprocessing for "
                    f"{len(pending)} pending results | batch_job_id={batch_job.id}"
                )
                _dispatch_tts_result_processing(run, batch_job, org_id, provider_name)
                return True

            poll_result = await poll_batch_jobs(
                session=session,
                batch_jobs=batch_jobs,
                batch_provider=batch_provider,
                provider_config_key="tts_provider",
                log_prefix=log_prefix,
                on_succeeded=_on_batch_succeeded,
                on_already_succeeded=_on_already_succeeded,
            )
            any_dispatched = poll_result.any_dispatched

            for batch_job in batch_jobs:
                if batch_job.provider_status not in TERMINAL_STATES:
                    continue
                if batch_job.provider_status == BatchJobState.SUCCEEDED.value:
                    continue
                _fail_pending_results(
                    session=session,
                    run_id=run.id,
                    model=batch_job.config.get("tts_provider", ""),
                    error_message=(
                        f"Gemini batch ended: "
                        f"{batch_job.error_message or batch_job.provider_status}"
                    ),
                )

    # Sync workers write rows as they go; past the stale window the worker is gone.
    if is_stale:
        for model in sync_models:
            _fail_pending_results(
                session=session,
                run_id=run.id,
                model=model,
                error_message="Synthesis did not complete",
            )

    finalized_run = finalize_tts_run_status(session=session, run_id=run.id)
    final_status = finalized_run.status if finalized_run else run.status
    error_message = finalized_run.error_message if finalized_run else None

    if final_status in ("completed", "failed"):
        action = final_status
    elif any_dispatched:
        action = "dispatched"
    else:
        action = "no_change"

    return make_poll_result(
        run=run,
        eval_type=EVAL_TYPE_LABEL,
        previous_status=previous_status,
        current_status=final_status,
        action=action,
        error=error_message,
    )


def _fail_pending_results(
    *, session: Session, run_id: int, model: str, error_message: str
) -> None:
    """Mark one model's PENDING rows of a run FAILED."""
    pending = get_pending_results_for_run(
        session=session, run_id=run_id, provider=model
    )
    if not pending:
        return
    logger.warning(
        f"[_fail_pending_results] Failing pending results | run_id={run_id}, "
        f"model={model}, count={len(pending)}, error={error_message}"
    )
    for result in pending:
        update_tts_result(
            session=session,
            result_id=result.id,
            status=JobStatus.FAILED.value,
            error_message=error_message,
        )
    session.commit()
