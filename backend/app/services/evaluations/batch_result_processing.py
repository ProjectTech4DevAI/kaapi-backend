"""Worker-side entry point for batch-mode text evaluation result processing.

Moves the heavy per-run download/parse/embedding/cosine work off the non-recycling
web process onto a Celery worker (prefork children recycle, reclaiming the
Langfuse-client memory the cron path used to leak). Dispatched per run by
`poll_all_pending_evaluations`; delegates the unchanged stage logic to
`check_and_process_evaluation`.
"""

import asyncio
import logging
from typing import Any

from fastapi import HTTPException
from sqlmodel import Session

from app.core.db import engine
from app.crud.evaluations.core import update_evaluation_run
from app.crud.evaluations.processing import check_and_process_evaluation
from app.models import EvaluationRun, EvaluationRunUpdate
from app.utils import get_openai_client, get_tracing_client

logger = logging.getLogger(__name__)


def execute_evaluation_batch_result_processing(
    *, project_id: int, eval_run_id: int, trace_id: str
) -> dict[str, Any] | None:
    """Process one batch-mode text EvaluationRun's completed results.

    Idempotent per the dispatch lease (the cron bumps ``updated_at`` before
    enqueueing) and ``check_and_process_evaluation``'s own stage branching: once
    ``embedding_batch_job_id`` is set, a redelivery takes the embedding branch
    rather than re-submitting the response batch, so it never re-charges OpenAI.

    On terminal failure the run is marked ``failed`` and the error re-raised, so a
    permanently broken run (e.g. bad org credentials) does not sit in ``processing``
    and get re-dispatched every stall window forever.
    """
    logger.info(
        f"[execute_evaluation_batch_result_processing] Starting | "
        f"project_id={project_id} | eval_run_id={eval_run_id} | trace_id={trace_id}"
    )

    with Session(engine) as session:
        eval_run = session.get(EvaluationRun, eval_run_id)
        if eval_run is None:
            logger.warning(
                f"[execute_evaluation_batch_result_processing] EvaluationRun not "
                f"found, nothing to process | eval_run_id={eval_run_id}"
            )
            return None

        langfuse = None
        try:
            openai_client = get_openai_client(
                session=session,
                org_id=eval_run.organization_id,
                project_id=project_id,
            )
            langfuse = get_tracing_client(
                session=session,
                org_id=eval_run.organization_id,
                project_id=project_id,
            )

            # check_and_process_evaluation is async (it awaits provider polling);
            # prefork workers run single-threaded, so a fresh loop per task is safe.
            result = asyncio.run(
                check_and_process_evaluation(
                    eval_run=eval_run,
                    session=session,
                    openai_client=openai_client,
                    langfuse=langfuse,
                )
            )
            logger.info(
                f"[execute_evaluation_batch_result_processing] Done | "
                f"eval_run_id={eval_run_id} | action={result.get('action')}"
            )
            return result

        except Exception as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            logger.error(
                f"[execute_evaluation_batch_result_processing] Run failed | "
                f"eval_run_id={eval_run_id} | error={detail}",
                exc_info=True,
            )
            # Re-fetch in case the failing call rolled our session back.
            session.rollback()
            failed_run = session.get(EvaluationRun, eval_run_id)
            if failed_run is not None:
                update_evaluation_run(
                    session=session,
                    eval_run=failed_run,
                    update=EvaluationRunUpdate(
                        status="failed",
                        error_message=f"Batch result processing failed: {detail}",
                    ),
                )
            raise

        finally:
            # Each Langfuse() spawns background TaskManager threads; flush before the
            # worker returns so ingestion completes and threads don't accumulate.
            if langfuse is not None:
                try:
                    langfuse.flush()
                except Exception as flush_err:
                    logger.warning(
                        f"[execute_evaluation_batch_result_processing] Langfuse "
                        f"flush failed | eval_run_id={eval_run_id} | {flush_err}"
                    )
