"""
Celery task for the synchronous (fast) text-evaluation pipeline.

This module hosts the single orchestrator task per fast evaluation run. The
heavy lifting lives in `app/services/evaluations/fast.py`; this task is a thin
shim that sets the correlation id, attaches the OTel parent context, and
delegates.

See `Fast Evaluation SRD.md` for the design (queue, retries, idempotency).
"""

import logging

from celery import current_task

from app.celery.celery_app import celery_app
from app.celery.tasks.job_execution import _run_with_otel_parent, _set_trace
from app.celery.utils import gevent_timeout
from app.core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, queue="evaluations", priority=6)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_evaluation_fast")
def run_evaluation_fast(
    self, eval_run_id: int, trace_id: str = "N/A", **kwargs
) -> None:
    """Run the fast evaluation pipeline for one EvaluationRun.

    Idempotency: each stage is skipped on retry when its `batch_job` marker is
    already set on the EvaluationRun, so Celery redelivery never re-calls
    OpenAI for work that already succeeded.

    Args:
        eval_run_id: ID of the EvaluationRun (run_mode="fast").
        trace_id: Correlation id from the enqueueing request, propagated into
            the worker for log correlation.
    """
    from app.services.evaluations.fast import execute_fast_evaluation

    _set_trace(trace_id)
    logger.info(
        f"[run_evaluation_fast] Starting fast evaluation task | "
        f"eval_run_id={eval_run_id} | task_id={current_task.request.id}"
    )

    return _run_with_otel_parent(
        self,
        lambda: execute_fast_evaluation(eval_run_id=eval_run_id),
    )
