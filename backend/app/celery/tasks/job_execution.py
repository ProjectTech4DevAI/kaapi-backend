import logging
from collections.abc import Callable
from celery import current_task
from asgi_correlation_id import correlation_id

from app.celery.celery_app import celery_app
import app.services.llm.jobs as _llm_jobs
import app.services.response.jobs as _response_jobs
import app.services.doctransform.job as _doctransform_job
import app.services.collections.create_collection as _create_collection
import app.services.collections.delete_collection as _delete_collection
import app.services.stt_evaluations.batch_job as _stt_batch_job
import app.services.stt_evaluations.metric_job as _stt_metric_job
import app.services.tts_evaluations.batch_job as _tts_batch_job
import app.services.tts_evaluations.batch_result_processing as _tts_result_processing

logger = logging.getLogger(__name__)

# Hardcoded dispatch table — avoids dynamic importlib at task execution time.
# Imports above happen once in the main Celery process before worker forks,
# so all child workers inherit them via copy-on-write instead of each loading
# them independently (which was causing OOM with warmup_job_modules).
_FUNCTION_REGISTRY: dict[str, Callable] = {
    "app.services.llm.jobs.execute_job": _llm_jobs.execute_job,
    "app.services.llm.jobs.execute_chain_job": _llm_jobs.execute_chain_job,
    "app.services.response.jobs.execute_job": _response_jobs.execute_job,
    "app.services.doctransform.job.execute_job": _doctransform_job.execute_job,
    "app.services.collections.create_collection.execute_job": _create_collection.execute_job,
    "app.services.collections.delete_collection.execute_job": _delete_collection.execute_job,
    "app.services.stt_evaluations.batch_job.execute_batch_submission": _stt_batch_job.execute_batch_submission,
    "app.services.stt_evaluations.metric_job.execute_metric_computation": _stt_metric_job.execute_metric_computation,
    "app.services.tts_evaluations.batch_job.execute_batch_submission": _tts_batch_job.execute_batch_submission,
    "app.services.tts_evaluations.batch_result_processing.execute_tts_result_processing": _tts_result_processing.execute_tts_result_processing,
}


@celery_app.task(bind=True, queue="high_priority")
def execute_high_priority_task(
    self,
    function_path: str,
    project_id: int,
    job_id: str,
    trace_id: str,
    **kwargs,
):
    """
    High priority Celery task to execute any job function.
    Use this for urgent operations that need immediate processing.

    Args:
        function_path: Import path to the execute_job function (e.g., "app.services.doctransform.service.execute_job")
        project_id: ID of the project executing the job
        job_id: ID of the job (should already exist in database)
        trace_id: Trace/correlation ID to preserve context across Celery tasks
        **kwargs: Additional arguments to pass to the execute_job function
    """
    return _execute_job_internal(
        self, function_path, project_id, job_id, "high_priority", trace_id, **kwargs
    )


@celery_app.task(bind=True, queue="low_priority")
def execute_low_priority_task(
    self,
    function_path: str,
    project_id: int,
    job_id: str,
    trace_id: str,
    **kwargs,
):
    """
    Low priority Celery task to execute any job function.
    Use this for background operations that can wait.

    Args:
        function_path: Import path to the execute_job function (e.g., "app.services.doctransform.service.execute_job")
        project_id: ID of the project executing the job
        job_id: ID of the job (should already exist in database)
        trace_id: Trace/correlation ID to preserve context across Celery tasks
        **kwargs: Additional arguments to pass to the execute_job function
    """
    return _execute_job_internal(
        self, function_path, project_id, job_id, "low_priority", trace_id, **kwargs
    )


def _execute_job_internal(
    task_instance,
    function_path: str,
    project_id: int,
    job_id: str,
    priority: str,
    trace_id: str,
    **kwargs,
):
    """
    Internal function to execute job logic for both priority levels.

    Args:
        task_instance: Celery task instance (for progress updates, retries, etc.)
        function_path: Import path to the execute_job function
        project_id: ID of the project executing the job
        job_id: ID of the job (should already exist in database)
        priority: Priority level ("high_priority" or "low_priority")
        trace_id: Trace/correlation ID to preserve context across Celery tasks
        **kwargs: Additional arguments to pass to the execute_job function
    """
    task_id = current_task.request.id

    correlation_id.set(trace_id)
    logger.info(f"Set correlation ID context: {trace_id} for job {job_id}")

    try:
        execute_function = _FUNCTION_REGISTRY.get(function_path)
        if execute_function is None:
            raise ValueError(
                f"[_execute_job_internal] Unknown function path: {function_path}"
            )

        logger.info(
            f"Executing {priority} job {job_id} (task {task_id}) using function {function_path}"
        )

        # Execute the business logic function with standardized parameters
        result = execute_function(
            project_id=project_id,
            job_id=job_id,
            task_id=task_id,
            task_instance=task_instance,  # For progress updates, retries if needed
            **kwargs,
        )

        logger.info(
            f"{priority.capitalize()} job {job_id} (task {task_id}) completed successfully"
        )
        return result

    except Exception as exc:
        logger.error(
            f"{priority.capitalize()} job {job_id} (task {task_id}) failed: {exc}",
            exc_info=True,
        )
        raise
