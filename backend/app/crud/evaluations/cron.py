"""
CRUD operations for evaluation cron jobs.

This module provides functions that can be invoked periodically to process
pending evaluations across all organizations.
"""

import asyncio
import logging
import math
from datetime import timedelta
from typing import Any

from sqlmodel import Session, select

from app.core.config import settings
from app.core.util import now
from app.crud.evaluations.core import update_evaluation_run
from app.crud.evaluations.fast import CHUNK_CONFIG_INDEX, list_response_chunk_jobs
from app.crud.evaluations.iteration import list_processing_evaluation_iteration_runs
from app.crud.evaluations.processing import poll_all_pending_evaluations
from app.models import EvaluationRun, EvaluationRunUpdate
from app.models.evaluation import RunModeEnum

logger = logging.getLogger(__name__)


def dispatch_fast_evaluation_barriers(session: Session) -> dict[str, Any]:
    """Fan-in barrier + stall healer for chunked fast evaluations.

    For every fast run still `processing`:
      * all chunks done and not yet aggregated (batch_job_id unset) → enqueue the
        aggregate task,
      * chunks missing and the run stalled past EVAL_FAST_STALL_THRESHOLD_MINUTES
        → re-enqueue the missing chunk indices (idempotent: a completed chunk is
        skipped, so this never re-charges OpenAI).

    ``batch_job_id`` (set by the aggregator's merge) is the double-enqueue guard:
    once merged, later ticks won't re-enqueue the aggregate, and a redelivered
    aggregate reloads the merged unit instead of re-merging.
    """
    from app.celery.utils import (
        start_fast_evaluation_aggregate,
        start_fast_evaluation_chunk,
    )

    statement = select(EvaluationRun).where(
        EvaluationRun.status == "processing",
        EvaluationRun.type == "text",
        EvaluationRun.run_mode == RunModeEnum.FAST.value,
    )
    runs = list(session.exec(statement).all())

    aggregates_dispatched = 0
    chunks_reenqueued = 0
    stall_cutoff = now() - timedelta(minutes=settings.EVAL_FAST_STALL_THRESHOLD_MINUTES)

    for run in runs:
        expected = math.ceil(max(run.total_items, 0) / settings.EVAL_FAST_CHUNK_SIZE)
        if expected == 0:
            continue

        done_indices = {
            int(job.config.get(CHUNK_CONFIG_INDEX, -1))
            for job in list_response_chunk_jobs(session=session, eval_run_id=run.id)
            if job.raw_output_url
        }
        done = len(done_indices)

        if done >= expected and run.batch_job_id is None:
            start_fast_evaluation_aggregate(eval_run_id=run.id)
            aggregates_dispatched += 1
            logger.info(
                f"[dispatch_fast_evaluation_barriers] Aggregate dispatched | "
                f"run_id={run.id} | chunks={done}/{expected}"
            )
            continue

        if done < expected and run.updated_at < stall_cutoff:
            missing = [i for i in range(expected) if i not in done_indices]
            for chunk_index in missing:
                start_fast_evaluation_chunk(eval_run_id=run.id, chunk_index=chunk_index)
            chunks_reenqueued += len(missing)
            # Bump updated_at so the stall window resets to the next tick's cadence.
            # ponytail: no hard retry budget — a permanently failing chunk keeps
            # re-enqueuing (cheap + idempotent). Add a retry-count field to fail
            # the run outright if provider outages must not linger.
            update_evaluation_run(
                session=session, eval_run=run, update=EvaluationRunUpdate()
            )
            logger.warning(
                f"[dispatch_fast_evaluation_barriers] Stalled run, re-enqueued "
                f"chunks | run_id={run.id} | missing={missing} | "
                f"done={done}/{expected}"
            )

    return {
        "total": len(runs),
        "aggregates_dispatched": aggregates_dispatched,
        "chunks_reenqueued": chunks_reenqueued,
    }


def dispatch_pending_evaluation_iteration_resumes(session: Session) -> dict[str, Any]:
    """Resume trigger for the eval-iterate-improve LangGraph loop.

    For every thin `evaluation_iteration_run` row still `PROCESSING`, dispatch a
    `resume=True` graph-step task. Cheap even when the loop is still waiting on
    the same sub-job as last tick: the task just re-checks status and interrupts
    again immediately if not ready — same cost profile as a plain polling barrier.
    """
    from app.celery.utils import start_evaluation_iteration_round

    runs = list_processing_evaluation_iteration_runs(session=session)
    for run in runs:
        start_evaluation_iteration_round(
            iteration_run_id=run.id,
            resume=True,
            organization_id=run.organization_id,
            project_id=run.project_id,
        )

    logger.info(
        f"[dispatch_pending_evaluation_iteration_resumes] Dispatched resumes | "
        f"count={len(runs)}"
    )
    return {"total": len(runs), "resumes_dispatched": len(runs)}


async def process_all_pending_evaluations(session: Session) -> dict[str, Any]:
    """
    Process all pending evaluations across all organizations.

    Delegates to poll_all_pending_evaluations which fetches all processing
    evaluation runs in a single query, groups by project, and processes them.
    Also polls STT and TTS evaluations similarly.

    Args:
        session: Database session

    Returns:
        Dict with aggregated results.
    """
    logger.info("[process_all_pending_evaluations] Starting evaluation processing")

    try:
        # Poll text evaluations (single query, grouped by project)
        text_summary = await poll_all_pending_evaluations(session=session)

        # Lazy imports to avoid circular dependency with cron_utils
        from app.crud.stt_evaluations import poll_all_pending_stt_evaluations
        from app.crud.tts_evaluations import poll_all_pending_tts_evaluations

        # Poll STT evaluations (single query, grouped by project)
        stt_summary = await poll_all_pending_stt_evaluations(session=session)

        # Poll TTS evaluations (single query, grouped by project)
        tts_summary = await poll_all_pending_tts_evaluations(session=session)

        # Fan-in barrier + stall healer for chunked fast-mode text evaluations.
        fast_summary = dispatch_fast_evaluation_barriers(session=session)

        # Resume trigger for the eval-iterate-improve LangGraph loop.
        iteration_summary = dispatch_pending_evaluation_iteration_resumes(
            session=session
        )

        # Merge summaries
        total_processed = (
            text_summary["processed"]
            + stt_summary["processed"]
            + tts_summary["processed"]
        )
        total_failed = (
            text_summary["failed"] + stt_summary["failed"] + tts_summary["failed"]
        )
        total_still_processing = (
            text_summary["still_processing"]
            + stt_summary["still_processing"]
            + tts_summary["still_processing"]
        )
        all_details = (
            text_summary.get("details", [])
            + stt_summary.get("details", [])
            + tts_summary.get("details", [])
        )

        logger.info(
            f"[process_all_pending_evaluations] Completed: "
            f"{total_processed} processed, {total_failed} failed, "
            f"{total_still_processing} still processing | "
            f"fast_aggregates={fast_summary['aggregates_dispatched']} | "
            f"fast_chunks_reenqueued={fast_summary['chunks_reenqueued']} | "
            f"iteration_resumes={iteration_summary['resumes_dispatched']}"
        )

        return {
            "status": "success",
            "total_processed": total_processed,
            "total_failed": total_failed,
            "total_still_processing": total_still_processing,
            "results": all_details,
            "fast": fast_summary,
            "iteration": iteration_summary,
        }

    except Exception as e:
        logger.error(
            f"[process_all_pending_evaluations] Fatal error: {e}",
            exc_info=True,
        )
        return {
            "status": "error",
            "total_processed": 0,
            "total_failed": 0,
            "total_still_processing": 0,
            "error": str(e),
            "results": [],
        }


def process_all_pending_evaluations_sync(session: Session) -> dict[str, Any]:
    """
    Synchronous wrapper for process_all_pending_evaluations.

    This function can be called from synchronous contexts (like FastAPI endpoints).

    Args:
        session: Database session

    Returns:
        Dict with aggregated results (same as process_all_pending_evaluations)
    """
    return asyncio.run(process_all_pending_evaluations(session=session))
