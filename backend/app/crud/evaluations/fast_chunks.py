"""`batch_job` bookkeeping for the fast evaluation stages.

A `batch_job` row carrying a `raw_output_url` is what marks a stage (or one
response chunk) as already done, so retries reload from S3 instead of re-calling
OpenAI. This module owns the queries and row shapes; the orchestrator in
`fast.py` decides when to write them.
"""

import logging
from typing import Any

from sqlalchemy import Integer
from sqlmodel import Session, select

from app.core.cloud.storage import CloudStorage
from app.crud.evaluations.fast_results import (
    EMBEDDING_USAGE_KEYS,
    RESPONSE_USAGE_KEYS,
    sum_usage,
)
from app.crud.job import create_batch_job, delete_batch_job
from app.models.batch_job import BatchJob, BatchJobCreate
from app.models.evaluation import EvaluationRun, RunModeEnum

logger = logging.getLogger(__name__)

# job_type discriminators on batch_job for the fast-path stages.
JOB_TYPE_EVALUATION_FAST = "evaluation_fast"
JOB_TYPE_EVALUATION_FAST_CHUNK = "evaluation_fast_chunk"
JOB_TYPE_EMBEDDING_FAST = "embedding_fast"

# batch_job.config keys tying a chunk row back to its run + slice.
CHUNK_CONFIG_RUN_ID = "eval_run_id"
CHUNK_CONFIG_INDEX = "chunk_index"

_RESPONSES_ENDPOINT = "/v1/responses"
_EMBEDDINGS_ENDPOINT = "/v1/embeddings"


def list_response_chunk_jobs(*, session: Session, eval_run_id: int) -> list[BatchJob]:
    """All response-chunk batch_jobs for a fast run, in any state."""
    statement = select(BatchJob).where(
        BatchJob.job_type == JOB_TYPE_EVALUATION_FAST_CHUNK,
        BatchJob.config[CHUNK_CONFIG_RUN_ID].astext.cast(Integer) == eval_run_id,
    )
    return list(session.exec(statement).all())


def get_chunk_job(
    *, session: Session, eval_run_id: int, chunk_index: int
) -> BatchJob | None:
    """The chunk batch_job for one (eval_run, chunk_index), or None."""
    statement = select(BatchJob).where(
        BatchJob.job_type == JOB_TYPE_EVALUATION_FAST_CHUNK,
        BatchJob.config[CHUNK_CONFIG_RUN_ID].astext.cast(Integer) == eval_run_id,
        BatchJob.config[CHUNK_CONFIG_INDEX].astext.cast(Integer) == chunk_index,
    )
    return session.exec(statement).first()


def _create_stage_job(
    *,
    session: Session,
    eval_run: EvaluationRun,
    job_type: str,
    config: dict[str, Any],
    raw_output_url: str | None,
    total_items: int,
) -> BatchJob:
    return create_batch_job(
        session=session,
        batch_job_create=BatchJobCreate(
            provider="openai",
            job_type=job_type,
            config={"run_mode": RunModeEnum.FAST.value, **config},
            raw_output_url=raw_output_url,
            total_items=total_items,
            organization_id=eval_run.organization_id,
            project_id=eval_run.project_id,
        ),
    )


def create_response_chunk_job(
    *,
    session: Session,
    eval_run: EvaluationRun,
    chunk_index: int,
    model: str | None,
    results: list[dict[str, Any]],
    raw_output_url: str | None,
) -> BatchJob:
    """Mark one response chunk done."""
    return _create_stage_job(
        session=session,
        eval_run=eval_run,
        job_type=JOB_TYPE_EVALUATION_FAST_CHUNK,
        config={
            "endpoint": _RESPONSES_ENDPOINT,
            "model": model,
            "usage": sum_usage(results, RESPONSE_USAGE_KEYS),
            CHUNK_CONFIG_RUN_ID: eval_run.id,
            CHUNK_CONFIG_INDEX: chunk_index,
        },
        raw_output_url=raw_output_url,
        total_items=len(results),
    )


def create_merged_response_job(
    *,
    session: Session,
    eval_run: EvaluationRun,
    model: str | None,
    results: list[dict[str, Any]],
    raw_output_url: str | None,
) -> BatchJob:
    """Mark the merged responses unit done; its id is the aggregate's retry guard."""
    return _create_stage_job(
        session=session,
        eval_run=eval_run,
        job_type=JOB_TYPE_EVALUATION_FAST,
        config={
            "endpoint": _RESPONSES_ENDPOINT,
            "model": model,
            "usage": sum_usage(results, RESPONSE_USAGE_KEYS),
        },
        raw_output_url=raw_output_url,
        total_items=len(results),
    )


def create_embedding_job(
    *,
    session: Session,
    eval_run: EvaluationRun,
    embedding_model: str,
    results: list[dict[str, Any]],
    raw_output_url: str | None,
) -> BatchJob:
    """Mark the embeddings unit done."""
    return _create_stage_job(
        session=session,
        eval_run=eval_run,
        job_type=JOB_TYPE_EMBEDDING_FAST,
        config={
            "endpoint": _EMBEDDINGS_ENDPOINT,
            "embedding_model": embedding_model,
            "usage": sum_usage(results, EMBEDDING_USAGE_KEYS),
        },
        raw_output_url=raw_output_url,
        total_items=len(results),
    )


def delete_response_chunk_artifacts(
    *, session: Session, storage: CloudStorage, eval_run_id: int
) -> None:
    """Delete the per-chunk S3 files + batch_job rows once a run completes.

    Best-effort: a failed delete only leaks DB+S3 bloat, so it never fails the
    run. Failed runs skip this and keep their chunks for the healer.
    """
    try:
        chunk_jobs = list_response_chunk_jobs(session=session, eval_run_id=eval_run_id)
        for job in chunk_jobs:
            if job.raw_output_url:
                storage.delete(job.raw_output_url)
            delete_batch_job(session, job)
        logger.info(
            f"[delete_response_chunk_artifacts] Removed {len(chunk_jobs)} chunk "
            f"artifacts | eval_run_id={eval_run_id}"
        )
    except Exception as exc:
        logger.warning(
            f"[delete_response_chunk_artifacts] Cleanup failed (orphans harmless) | "
            f"eval_run_id={eval_run_id} | error={exc}",
            exc_info=True,
        )
