"""Celery tasks for live (non-batch) evaluation mode.

Two tasks form a chord:

* `run_eval_row` — one task per dataset item. Calls the OpenAI Responses
  API and Embeddings API and returns a result dict. Retries on transient
  OpenAI errors (rate limit, timeout, connection); permanent errors are
  caught and folded into the result dict so the chord proceeds.

* `aggregate_eval_results` — chord callback. Idempotent. Writes Langfuse
  traces, computes cosine similarity, attaches cost, marks the eval run
  completed.

Both run on the dedicated `evaluations` queue so they don't compete with
user-facing `high_priority` traffic.
"""

import logging

import openai
from celery import current_task

from app.celery.celery_app import celery_app
from app.celery.tasks.job_execution import _run_with_otel_parent, _set_trace

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    queue="evaluations",
    priority=5,
    autoretry_for=(
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
    ),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=5,
    soft_time_limit=120,
    time_limit=180,
)
def run_eval_row(
    self,
    eval_run_id: int,
    item: dict,
    organization_id: int,
    project_id: int,
    config_id: str,
    config_version: int,
    trace_id: str = "N/A",
):
    from app.services.evaluations.live_row import execute_eval_row

    _set_trace(trace_id)
    logger.info(
        f"[run_eval_row] task_id={current_task.request.id} | eval={eval_run_id} | "
        f"item={item.get('id')}"
    )
    return _run_with_otel_parent(
        self,
        lambda: execute_eval_row(
            eval_run_id=eval_run_id,
            item=item,
            organization_id=organization_id,
            project_id=project_id,
            config_id=config_id,
            config_version=config_version,
        ),
    )


@celery_app.task(
    bind=True,
    queue="evaluations",
    priority=6,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    soft_time_limit=240,
    time_limit=300,
)
def aggregate_eval_results(
    self,
    row_results: list[dict],
    eval_run_id: int,
    organization_id: int,
    project_id: int,
    trace_id: str = "N/A",
):
    from app.services.evaluations.live_aggregator import aggregate_results

    _set_trace(trace_id)
    logger.info(
        f"[aggregate_eval_results] task_id={current_task.request.id} | "
        f"eval={eval_run_id} | rows={len(row_results)}"
    )
    return _run_with_otel_parent(
        self,
        lambda: aggregate_results(
            eval_run_id=eval_run_id,
            organization_id=organization_id,
            project_id=project_id,
            row_results=row_results,
        ),
    )
