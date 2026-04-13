import logging

from asgi_correlation_id import correlation_id
from celery import current_task

from app.celery.celery_app import celery_app

logger = logging.getLogger(__name__)


def _set_trace(trace_id: str) -> None:
    correlation_id.set(trace_id)
    logger.info(f"[_set_trace] Set correlation ID: {trace_id}")


@celery_app.task(bind=True, queue="high_priority", priority=9)
def run_llm_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.llm.jobs import execute_job

    _set_trace(trace_id)
    return execute_job(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )


@celery_app.task(bind=True, queue="high_priority", priority=9)
def run_llm_chain_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.llm.jobs import execute_chain_job

    _set_trace(trace_id)
    return execute_chain_job(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )


@celery_app.task(bind=True, queue="high_priority", priority=9)
def run_response_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.response.jobs import execute_job

    _set_trace(trace_id)
    return execute_job(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )


@celery_app.task(bind=True, queue="low_priority", priority=1)
def run_doctransform_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.doctransform.job import execute_job

    _set_trace(trace_id)
    return execute_job(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )


@celery_app.task(bind=True, queue="low_priority", priority=1)
def run_create_collection_job(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.collections.create_collection import execute_job

    _set_trace(trace_id)
    return execute_job(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )


@celery_app.task(bind=True, queue="low_priority", priority=1)
def run_delete_collection_job(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.collections.delete_collection import execute_job

    _set_trace(trace_id)
    return execute_job(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )


@celery_app.task(bind=True, queue="low_priority", priority=1)
def run_stt_batch_submission(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.stt_evaluations.batch_job import execute_batch_submission

    _set_trace(trace_id)
    return execute_batch_submission(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )


@celery_app.task(bind=True, queue="low_priority", priority=1)
def run_stt_metric_computation(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.stt_evaluations.metric_job import execute_metric_computation

    _set_trace(trace_id)
    return execute_metric_computation(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )


@celery_app.task(bind=True, queue="low_priority", priority=1)
def run_tts_batch_submission(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.tts_evaluations.batch_job import execute_batch_submission

    _set_trace(trace_id)
    return execute_batch_submission(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )


@celery_app.task(bind=True, queue="low_priority", priority=1)
def run_tts_result_processing(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.tts_evaluations.batch_result_processing import (
        execute_tts_result_processing,
    )

    _set_trace(trace_id)
    return execute_tts_result_processing(
        project_id=project_id,
        job_id=job_id,
        task_id=current_task.request.id,
        task_instance=self,
        **kwargs,
    )
