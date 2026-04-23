import logging

from asgi_correlation_id import correlation_id
from celery import current_task
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract

from app.celery.celery_app import celery_app

logger = logging.getLogger(__name__)


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


@celery_app.task(bind=True, queue="high_priority", priority=9)
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


@celery_app.task(bind=True, queue="high_priority", priority=9)
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


@celery_app.task(bind=True, queue="high_priority", priority=9)
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


@celery_app.task(bind=True, queue="low_priority", priority=1)
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


@celery_app.task(bind=True, queue="low_priority", priority=1)
def run_create_collection_job(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.collections.create_collection import execute_job

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


@celery_app.task(bind=True, queue="low_priority", priority=1)
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


@celery_app.task(bind=True, queue="low_priority", priority=1)
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


@celery_app.task(bind=True, queue="low_priority", priority=1)
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


@celery_app.task(bind=True, queue="low_priority", priority=1)
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


@celery_app.task(bind=True, queue="low_priority", priority=1)
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
