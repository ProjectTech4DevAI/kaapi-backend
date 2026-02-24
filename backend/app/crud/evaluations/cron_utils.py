"""Shared utilities for evaluation cron processing.

Common constants, queries, and helpers used by both STT and TTS
evaluation polling loops.
"""

from collections import defaultdict

from sqlalchemy import Integer
from sqlmodel import Session, select

from app.core.batch import BatchJobState
from app.models import EvaluationRun
from app.models.batch_job import BatchJob

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
