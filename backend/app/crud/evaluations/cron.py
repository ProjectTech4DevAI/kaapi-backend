"""Periodic evaluation maintenance: polling, fast-eval barriers, iteration resumes."""

import asyncio
import logging
import math
from datetime import timedelta
from typing import Any

from sqlmodel import Session, select

from app.core.config import settings
from app.core.util import now
from app.crud.evaluations.core import update_evaluation_run
from app.crud.evaluations.fast_chunks import (
    CHUNK_CONFIG_INDEX,
    list_response_chunk_jobs,
)
from app.crud.evaluations.iteration import (
    list_processing_evaluation_iteration_runs,
    update_evaluation_iteration_run,
)
from app.crud.evaluations.processing import poll_all_pending_evaluations
from app.models import EvaluationRun, EvaluationRunUpdate
from app.models.evaluation import RunModeEnum
from app.models.evaluation_iteration import EvaluationIterationRunUpdate

logger = logging.getLogger(__name__)


def dispatch_fast_evaluation_barriers(session: Session) -> dict[str, Any]:
    """Fan-in barrier + stall healer for chunked fast evaluations.

    All chunks done → enqueue aggregate (`batch_job_id`, set by the merge, guards
    against double-enqueue). Stalled with chunks missing → re-enqueue those
    indices (idempotent, completed chunks are skipped).
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
    """Resume every PROCESSING iteration loop; fail those stalled past the threshold.

    Loops dispatched within the cooldown are skipped so a slow step never races a
    second one on the same checkpoint thread.
    """
    from app.celery.utils import start_evaluation_iteration_round
    from app.services.evaluations.iteration_graph import mark_iteration_run_failed

    runs = list_processing_evaluation_iteration_runs(session=session)
    current_time = now()
    stall_cutoff = current_time - timedelta(
        hours=settings.EVAL_ITERATION_STALL_THRESHOLD_HOURS
    )
    cooldown_cutoff = current_time - timedelta(
        minutes=settings.EVAL_ITERATION_DISPATCH_COOLDOWN_MINUTES
    )

    dispatched = 0
    reaped = 0
    in_flight = 0

    for run in runs:
        if run.inserted_at < stall_cutoff:
            logger.warning(
                f"[dispatch_pending_evaluation_iteration_resumes] Reaping stalled "
                f"loop | iteration_run_id={run.id} | inserted_at={run.inserted_at}"
            )
            mark_iteration_run_failed(
                iteration_run_id=run.id,
                organization_id=run.organization_id,
                project_id=run.project_id,
                error_message=(
                    f"Evaluation iteration loop stalled: still processing more than "
                    f"{settings.EVAL_ITERATION_STALL_THRESHOLD_HOURS}h after it started."
                ),
            )
            reaped += 1
            continue

        if (
            run.last_dispatched_at is not None
            and run.last_dispatched_at > cooldown_cutoff
        ):
            in_flight += 1
            continue

        start_evaluation_iteration_round(
            iteration_run_id=run.id,
            resume=True,
            organization_id=run.organization_id,
            project_id=run.project_id,
        )
        # Stamp after enqueue so a failed dispatch retries next tick.
        update_evaluation_iteration_run(
            session=session,
            iteration_run=run,
            update=EvaluationIterationRunUpdate(last_dispatched_at=current_time),
        )
        dispatched += 1

    logger.info(
        f"[dispatch_pending_evaluation_iteration_resumes] Dispatched resumes | "
        f"count={dispatched} | in_flight_skipped={in_flight} | reaped={reaped}"
    )
    return {
        "total": len(runs),
        "resumes_dispatched": dispatched,
        "in_flight_skipped": in_flight,
        "reaped": reaped,
    }


async def process_all_pending_evaluations(session: Session) -> dict[str, Any]:
    """Poll text/STT/TTS evaluations, then run the fast-eval barrier and iteration resumes."""
    logger.info("[process_all_pending_evaluations] Starting evaluation processing")

    try:
        text_summary = await poll_all_pending_evaluations(session=session)

        from app.crud.stt_evaluations import poll_all_pending_stt_evaluations
        from app.crud.tts_evaluations import poll_all_pending_tts_evaluations

        stt_summary = await poll_all_pending_stt_evaluations(session=session)
        tts_summary = await poll_all_pending_tts_evaluations(session=session)
        fast_summary = dispatch_fast_evaluation_barriers(session=session)
        iteration_summary = dispatch_pending_evaluation_iteration_resumes(
            session=session
        )

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
    """Sync wrapper for FastAPI endpoints."""
    return asyncio.run(process_all_pending_evaluations(session=session))
