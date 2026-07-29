"""Shared utilities for evaluation cron processing.

Common constants, queries, and helpers used by both STT and TTS
evaluation polling loops.
"""

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Integer
from sqlmodel import Session, select

from app.core.batch import (
    BatchJobState,
    GeminiBatchProvider,
    GeminiClient,
    poll_batch_status,
)
from app.models import EvaluationRun
from app.models.batch_job import BatchJob

logger = logging.getLogger(__name__)

# Terminal states that indicate batch processing is complete
TERMINAL_STATES = {
    BatchJobState.SUCCEEDED.value,
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
}


def fetch_processing_runs(
    session: Session,
    eval_type: str,
) -> list[EvaluationRun]:
    """Fetch all evaluation runs with status='processing' for a given type.

    Args:
        session: Database session
        eval_type: Evaluation type value (e.g. EvaluationType.STT.value)

    Returns:
        list[EvaluationRun]: Runs currently processing
    """
    statement = select(EvaluationRun).where(
        EvaluationRun.type == eval_type,
        EvaluationRun.status == "processing",
        EvaluationRun.batch_job_id.is_not(None),
    )
    return list(session.exec(statement).all())


def group_runs_by_project(
    runs: list[EvaluationRun],
) -> dict[int, list[EvaluationRun]]:
    """Group evaluation runs by project_id.

    Args:
        runs: List of evaluation runs

    Returns:
        dict mapping project_id to list of runs
    """
    by_project: dict[int, list[EvaluationRun]] = defaultdict(list)
    for run in runs:
        by_project[run.project_id].append(run)
    return by_project


def get_batch_jobs_for_run(
    session: Session,
    run: EvaluationRun,
    job_type: str,
) -> list[BatchJob]:
    """Find all batch jobs associated with an evaluation run.

    Args:
        session: Database session
        run: The evaluation run
        job_type: Batch job type (e.g. "stt_evaluation", "tts_evaluation")

    Returns:
        list[BatchJob]: All batch jobs for this run
    """
    stmt = select(BatchJob).where(
        BatchJob.job_type == job_type,
        BatchJob.config["evaluation_run_id"].astext.cast(Integer) == run.id,
    )
    return list(session.exec(stmt).all())


def make_empty_summary() -> dict:
    """Return an empty polling summary."""
    return {
        "total": 0,
        "processed": 0,
        "failed": 0,
        "still_processing": 0,
        "details": [],
    }


def make_failure_result(
    run: EvaluationRun,
    eval_type: str,
    error: str,
) -> dict:
    """Build a failure result dict for a run.

    Args:
        run: The evaluation run
        eval_type: Short type label ("stt" or "tts")
        error: Error message

    Returns:
        dict with run_id, run_name, type, action, and error
    """
    return {
        "run_id": run.id,
        "run_name": run.run_name,
        "type": eval_type,
        "action": "failed",
        "error": error,
    }


def make_poll_result(
    run: EvaluationRun,
    eval_type: str,
    previous_status: str,
    current_status: str,
    action: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a status result dict for a polled run.

    Args:
        run: The evaluation run
        eval_type: Short type label ("stt" or "tts")
        previous_status: Status before polling
        current_status: Status after polling
        action: Action taken (e.g. "completed", "failed", "no_change")
        error: Optional error message

    Returns:
        dict with run_id, run_name, type, statuses, action, and optional error
    """
    result: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.run_name,
        "type": eval_type,
        "previous_status": previous_status,
        "current_status": current_status,
        "action": action,
    }
    if error:
        result["error"] = error
    return result


@dataclass
class BatchPollResult:
    """Aggregated result of polling all batch jobs for a single run."""

    all_terminal: bool
    any_succeeded: bool
    any_failed: bool
    any_dispatched: bool
    errors: list[str]


async def poll_batch_jobs(
    session: Session,
    batch_jobs: list[BatchJob],
    batch_provider: GeminiBatchProvider,
    provider_config_key: str,
    log_prefix: str,
    on_succeeded: Callable[[BatchJob, str], Awaitable[bool]],
    on_already_succeeded: Callable[[BatchJob, str], Awaitable[bool]] | None = None,
) -> BatchPollResult:
    """Poll a list of batch jobs and invoke callbacks on terminal states.

    Iterates over batch jobs, polls non-terminal ones, and calls the
    appropriate callback when a batch reaches or is already in SUCCEEDED state.

    Args:
        session: Database session
        batch_jobs: Batch jobs to poll
        batch_provider: Initialized GeminiBatchProvider
        provider_config_key: Config key for provider name (e.g. "stt_provider")
        log_prefix: Prefix for log messages
        on_succeeded: Called when a batch newly reaches SUCCEEDED.
            Returns True if async work was dispatched.
        on_already_succeeded: Called when a batch was already SUCCEEDED
            (e.g. for retry logic). Returns True if work was dispatched.
            Pass None to skip.

    Returns:
        BatchPollResult with aggregated state
    """
    all_terminal = True
    any_succeeded = False
    any_failed = False
    any_dispatched = False
    errors: list[str] = []

    for batch_job in batch_jobs:
        provider_name = batch_job.config.get(provider_config_key, "unknown")

        # Handle batch jobs already in terminal state
        if batch_job.provider_status in TERMINAL_STATES:
            if batch_job.provider_status == BatchJobState.SUCCEEDED.value:
                if on_already_succeeded:
                    if await on_already_succeeded(batch_job, provider_name):
                        any_dispatched = True
                any_succeeded = True
            else:
                any_failed = True
                errors.append(
                    f"{provider_name}: "
                    f"{batch_job.error_message or batch_job.provider_status}"
                )
            continue

        # Poll batch job status
        poll_batch_status(
            session=session,
            provider=batch_provider,
            batch_job=batch_job,
        )

        session.refresh(batch_job)
        provider_status = batch_job.provider_status

        logger.info(
            f"{log_prefix} Batch status | "
            f"batch_job_id={batch_job.id} | provider={provider_name} | "
            f"state={provider_status}"
        )

        if provider_status not in TERMINAL_STATES:
            all_terminal = False
            continue

        # Batch reached terminal state
        if provider_status == BatchJobState.SUCCEEDED.value:
            if await on_succeeded(batch_job, provider_name):
                any_dispatched = True
            any_succeeded = True
        else:
            any_failed = True
            errors.append(
                f"{provider_name}: {batch_job.error_message or provider_status}"
            )

    return BatchPollResult(
        all_terminal=all_terminal,
        any_succeeded=any_succeeded,
        any_failed=any_failed,
        any_dispatched=any_dispatched,
        errors=errors,
    )


async def poll_all_pending_evaluations_by_type(
    session: Session,
    *,
    eval_type: str,
    eval_type_enum_value: str,
    update_run_fn: Callable[..., Any],
    poll_run_fn: Callable[..., Awaitable[dict[str, Any]]],
    success_actions: tuple[str, ...] = ("completed", "processed"),
) -> dict[str, Any]:
    """Generic polling loop for pending evaluations of a given type.

    Fetches all runs with status='processing', groups by project,
    initializes a Gemini client per project, and polls each run.

    Args:
        session: Database session
        eval_type: Short type label for logging ("stt" or "tts")
        eval_type_enum_value: EvaluationType enum value for DB queries
        update_run_fn: Function to update a run's status/error
        poll_run_fn: Async function to poll a single run
        success_actions: Action strings that count as "processed"

    Returns:
        Summary dict with total, processed, failed, still_processing counts
    """
    fn_name = f"poll_all_pending_{eval_type}_evaluations"
    logger.info(f"[{fn_name}] Starting {eval_type.upper()} evaluation polling")

    pending_runs = fetch_processing_runs(session, eval_type_enum_value)

    if not pending_runs:
        logger.info(f"[{fn_name}] No pending {eval_type.upper()} runs found")
        return make_empty_summary()

    logger.info(
        f"[{fn_name}] Found {len(pending_runs)} pending {eval_type.upper()} runs"
    )

    evaluations_by_project = group_runs_by_project(pending_runs)

    all_results: list[dict[str, Any]] = []
    total_processed = 0
    total_failed = 0
    total_still_processing = 0

    for project_id, project_runs in evaluations_by_project.items():
        org_id = project_runs[0].organization_id

        try:
            try:
                gemini_client = GeminiClient.from_credentials(
                    session=session,
                    org_id=org_id,
                    project_id=project_id,
                )
            except Exception as client_err:
                logger.error(
                    f"[{fn_name}] Failed to get Gemini client | "
                    f"org_id={org_id} | project_id={project_id} | error={client_err}",
                    exc_info=True,
                )
                for run in project_runs:
                    update_run_fn(
                        session=session,
                        run_id=run.id,
                        status="failed",
                        error_message=f"Gemini client initialization failed: {str(client_err)}",
                    )
                    all_results.append(
                        make_failure_result(run, eval_type, str(client_err))
                    )
                    total_failed += 1
                continue

            batch_provider = GeminiBatchProvider(client=gemini_client.client)

            for run in project_runs:
                try:
                    result = await poll_run_fn(
                        session=session,
                        run=run,
                        batch_provider=batch_provider,
                        org_id=org_id,
                    )
                    all_results.append(result)

                    if result["action"] in success_actions:
                        total_processed += 1
                    elif result["action"] == "failed":
                        total_failed += 1
                    else:
                        total_still_processing += 1

                except Exception as e:
                    logger.error(
                        f"[{fn_name}] Failed to poll {eval_type.upper()} run | "
                        f"run_id={run.id} | {e}",
                        exc_info=True,
                    )
                    update_run_fn(
                        session=session,
                        run_id=run.id,
                        status="failed",
                        error_message=f"Polling failed: {str(e)}",
                    )
                    all_results.append(make_failure_result(run, eval_type, str(e)))
                    total_failed += 1

        except Exception as e:
            logger.error(
                f"[{fn_name}] Failed to process project | "
                f"project_id={project_id} | {e}",
                exc_info=True,
            )
            for run in project_runs:
                update_run_fn(
                    session=session,
                    run_id=run.id,
                    status="failed",
                    error_message=f"Project processing failed: {str(e)}",
                )
                all_results.append(
                    make_failure_result(
                        run, eval_type, f"Project processing failed: {str(e)}"
                    )
                )
                total_failed += 1

    summary = {
        "total": len(pending_runs),
        "processed": total_processed,
        "failed": total_failed,
        "still_processing": total_still_processing,
        "details": all_results,
    }

    logger.info(
        f"[{fn_name}] Polling summary | "
        f"processed={total_processed} | failed={total_failed} | "
        f"still_processing={total_still_processing}"
    )

    return summary
