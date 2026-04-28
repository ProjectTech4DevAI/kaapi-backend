"""Live evaluation orchestration: fan out per-row Celery tasks via a chord.

This is the live counterpart to `start_evaluation_batch()`. It fetches dataset
items from Langfuse, transitions the eval run to "processing", and dispatches
a Celery chord whose body is the aggregator task.
"""

import logging
from uuid import UUID

from celery import chord
from langfuse import Langfuse
from opentelemetry.propagate import inject
from sqlmodel import Session

from app.crud.evaluations.batch import fetch_dataset_items
from app.models import EvaluationRun

logger = logging.getLogger(__name__)


def _trace_headers() -> dict[str, str]:
    """Build OTel propagation headers for fan-out tasks."""
    headers: dict[str, str] = {}
    inject(headers)
    out = dict(headers)
    out["otel"] = headers
    return out


def start_evaluation_live(
    *,
    session: Session,
    eval_run: EvaluationRun,
    config_id: UUID,
    config_version: int,
    langfuse: Langfuse,
) -> EvaluationRun:
    """Fetch dataset items, mark run as processing, dispatch the Celery chord.

    Mirrors the shape of `start_evaluation_batch()` but builds no JSONL — each
    row runs as its own Celery task. Failures during dispatch mark the run
    as `failed` and re-raise.
    """
    log_prefix = (
        f"[start_evaluation_live][org={eval_run.organization_id}]"
        f"[project={eval_run.project_id}][eval={eval_run.id}]"
    )
    logger.info(f"{log_prefix} Starting live evaluation | run={eval_run.run_name}")

    try:
        dataset_items = fetch_dataset_items(
            langfuse=langfuse, dataset_name=eval_run.dataset_name
        )
        if not dataset_items:
            raise ValueError("Dataset is empty")

        # Import here to avoid pulling Celery into call paths that don't need it.
        from app.celery.tasks.evaluation_live import (
            aggregate_eval_results,
            run_eval_row,
        )

        eval_run.status = "processing"
        eval_run.total_items = len(dataset_items)
        session.add(eval_run)
        session.commit()
        session.refresh(eval_run)

        otel_headers = _trace_headers()

        row_signatures = [
            run_eval_row.s(
                eval_run_id=eval_run.id,
                item=item,
                organization_id=eval_run.organization_id,
                project_id=eval_run.project_id,
                config_id=str(config_id),
                config_version=config_version,
            ).set(headers=otel_headers)
            for item in dataset_items
        ]

        callback = aggregate_eval_results.s(
            eval_run_id=eval_run.id,
            organization_id=eval_run.organization_id,
            project_id=eval_run.project_id,
        ).set(headers=otel_headers)

        async_result = chord(row_signatures)(callback)

        logger.info(
            f"{log_prefix} Dispatched live chord | items={len(dataset_items)} | "
            f"chord_id={async_result.id}"
        )
        return eval_run

    except Exception as e:
        logger.error(
            f"{log_prefix} Failed to start live evaluation | {e}", exc_info=True
        )
        eval_run.status = "failed"
        eval_run.error_message = str(e)
        session.add(eval_run)
        session.commit()
        raise
