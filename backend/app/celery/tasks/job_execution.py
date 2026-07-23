"""Celery task definitions for the single priority `default` queue.

All tasks share one queue (`default`, declared with `x-max-priority=10`) and are
ordered by the per-task `priority`:

    9  LLM call + LLM chain (run_llm_job, run_llm_chain_job, run_response_job)
    6  Fast evaluation (run_evaluation_fast_chunk, run_evaluation_fast_aggregate)
    2  Everything else (doctransform, collections, STT/TTS evaluation, assessment)
    1  Notifications (send_eval_completion_notification)

Higher priority drains first; within the same priority, delivery is FIFO.
"""

import logging

from asgi_correlation_id import correlation_id
from celery import Task, current_task
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract

from app.celery.celery_app import celery_app
from app.celery.utils import gevent_timeout
from app.core.config import settings

logger = logging.getLogger(__name__)

# Sentinel correlation id used when no trace id is propagated from the
# enqueueing request. Matches the codebase-wide "N/A" default (see
# app/core/logger.py and app/celery/utils.py).
DEFAULT_TRACE_ID = "N/A"


def _set_trace(trace_id: str) -> None:
    correlation_id.set(trace_id)
    logger.info(f"[_set_trace] Set correlation ID: {trace_id}")


def _extract_parent_context(task_instance) -> otel_context.Context:
    """Extract OTel parent context from Celery headers if available."""
    headers = getattr(task_instance.request, "headers", None) or {}
    carrier: dict[str, str] = {}

    if isinstance(headers, dict):
        for key, value in headers.items():
            if isinstance(value, str):
                carrier[str(key)] = value

        nested = headers.get("otel", {})
        if isinstance(nested, dict):
            for key, value in nested.items():
                if isinstance(value, str):
                    carrier[str(key)] = value

    return extract(carrier)


def _run_with_otel_parent(task_instance, fn):
    """Attach extracted parent context and execute function.

    When Celery auto-instrumentation is active, there is already a current
    `run/...` span. Re-attaching extracted parent context here would make
    service spans become siblings of `run/...` instead of children.

    We only attach extracted context as a fallback when no active span exists.
    """
    current_ctx = trace.get_current_span().get_span_context()
    if current_ctx and current_ctx.is_valid:
        return fn()

    parent_ctx = _extract_parent_context(task_instance)
    token = otel_context.attach(parent_ctx)
    try:
        return fn()
    finally:
        otel_context.detach(token)


@celery_app.task(bind=True, queue="default", priority=9)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_llm_job")
def run_llm_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.llm.jobs import execute_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=9)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_llm_chain_job")
def run_llm_chain_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.llm.jobs import execute_chain_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_chain_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=9)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_response_job")
def run_response_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.response.jobs import execute_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=9)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_guardrails_job")
def run_guardrails_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.guardrails.jobs import execute_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_doctransform_job")
def run_doctransform_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.doctransform.job import execute_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_collection_setup_job")
def run_collection_setup_job(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.collections.create_collection import execute_setup_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_setup_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_collection_batch_job")
def run_collection_batch_job(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.collections.create_collection import execute_batch_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_batch_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_delete_collection_job")
def run_delete_collection_job(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.collections.delete_collection import execute_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_evaluation_batch_submission")
def run_evaluation_batch_submission(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.evaluations.batch_job import execute_evaluation_batch_submission

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_evaluation_batch_submission(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


# Priority 6 (fast-eval tier): user-blocking interactive prompt iteration in the
# evaluation domain, above default batch work but below core LLM call/chain jobs.
@celery_app.task(bind=True, queue="default", priority=6)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_prompt_improvement")
def run_prompt_improvement(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.evaluations.prompt_improvement import execute_prompt_improvement

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_prompt_improvement(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_stt_batch_submission")
def run_stt_batch_submission(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.stt_evaluations.batch_job import execute_batch_submission

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_batch_submission(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_stt_metric_computation")
def run_stt_metric_computation(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.stt_evaluations.metric_job import execute_metric_computation

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_metric_computation(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_tts_batch_submission")
def run_tts_batch_submission(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.tts_evaluations.batch_job import execute_batch_submission

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_batch_submission(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_assessment_pipeline")
def run_assessment_pipeline(
    self,
    run_id: int,
    organization_id: int,
    project_id: int,
    trace_id: str,
    **kwargs,
):
    from app.services.assessment.tasks import execute_assessment_pipeline

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_assessment_pipeline(
            run_id=run_id,
            organization_id=organization_id,
            project_id=project_id,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_tts_result_processing")
def run_tts_result_processing(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.tts_evaluations.batch_result_processing import (
        execute_tts_result_processing,
    )

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_tts_result_processing(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=6)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_evaluation_fast_chunk")
def run_evaluation_fast_chunk(
    self: Task,
    eval_run_id: int,
    chunk_index: int,
    trace_id: str = DEFAULT_TRACE_ID,
) -> None:
    """Run one responses chunk of a fast EvaluationRun.

    Idempotent: the chunk is skipped when its (eval_run, chunk_index) batch_job
    already has a raw_output_url, so Celery redelivery never re-charges OpenAI.
    """
    from app.services.evaluations.fast import execute_fast_evaluation_chunk

    _set_trace(trace_id)
    logger.info(
        f"[run_evaluation_fast_chunk] Starting fast eval chunk | "
        f"eval_run_id={eval_run_id} | chunk_index={chunk_index} | "
        f"task_id={current_task.request.id}"
    )

    return _run_with_otel_parent(
        self,
        lambda: execute_fast_evaluation_chunk(
            eval_run_id=eval_run_id, chunk_index=chunk_index
        ),
    )


@celery_app.task(bind=True, queue="default", priority=6)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_evaluation_fast_aggregate")
def run_evaluation_fast_aggregate(
    self: Task, eval_run_id: int, trace_id: str = DEFAULT_TRACE_ID
) -> None:
    """Fan-in: merge the response chunks then run embeddings + scoring.

    Enqueued once by the cron barrier after every chunk has completed.
    Idempotent: each stage is skipped on retry when its `batch_job` marker is
    set, so redelivery never re-does completed work.
    """
    from app.services.evaluations.fast import execute_fast_evaluation_aggregate

    _set_trace(trace_id)
    logger.info(
        f"[run_evaluation_fast_aggregate] Starting fast eval aggregate | "
        f"eval_run_id={eval_run_id} | task_id={current_task.request.id}"
    )

    return _run_with_otel_parent(
        self,
        lambda: execute_fast_evaluation_aggregate(eval_run_id=eval_run_id),
    )


@celery_app.task(bind=True, queue="default", priority=1)
@gevent_timeout(
    settings.CELERY_TASK_SOFT_TIME_LIMIT, "send_eval_completion_notification"
)
def send_eval_completion_notification(self, evaluation_id: int) -> dict:
    """
    Fan out a completion notification for an eval run to every project member.

    Idempotency: the `notification` table acts as the guard — see
    `app.services.notifications.eval_completion.execute_eval_completion_notification`
    for the full flow (it bails out if rows already exist for this
    entity_type/entity_id/notification_type).
    """
    from app.services.notifications.eval_completion import (
        execute_eval_completion_notification,
    )

    return execute_eval_completion_notification(evaluation_id=evaluation_id)
