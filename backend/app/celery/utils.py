"""
Utility functions for easy Celery integration across the application.
Business logic modules can use these functions without knowing Celery internals.
"""
import logging
import functools
from typing import Any, Dict, TypeVar
from collections.abc import Callable

from celery.result import AsyncResult
from gevent import Timeout
from opentelemetry.propagate import inject

from app.celery.celery_app import celery_app

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _enqueue_with_trace_context(task, **kwargs) -> str:
    """Publish Celery task with explicit trace context headers."""
    otel_headers: dict[str, str] = {}
    inject(otel_headers)
    celery_headers = dict(otel_headers)
    celery_headers["otel"] = otel_headers
    async_result = task.apply_async(kwargs=kwargs, headers=celery_headers)
    return async_result.id


def start_llm_job(project_id: int, job_id: str, trace_id: str = "N/A", **kwargs) -> str:
    from app.celery.tasks.job_execution import run_llm_job

    task_id = _enqueue_with_trace_context(
        run_llm_job, project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(f"[start_llm_job] Started job {job_id} with Celery task {task_id}")
    return task_id


def start_llm_chain_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_llm_chain_job

    task_id = _enqueue_with_trace_context(
        run_llm_chain_job,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(
        f"[start_llm_chain_job] Started job {job_id} with Celery task {task_id}"
    )
    return task_id


def start_response_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_response_job

    task_id = _enqueue_with_trace_context(
        run_response_job,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(f"[start_response_job] Started job {job_id} with Celery task {task_id}")
    return task_id


def start_doctransform_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_doctransform_job

    task_id = _enqueue_with_trace_context(
        run_doctransform_job,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(
        f"[start_doctransform_job] Started job {job_id} with Celery task {task_id}"
    )
    return task_id


def start_collection_setup_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_collection_setup_job

    task_id = _enqueue_with_trace_context(
        run_collection_setup_job,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(
        f"[start_collection_setup_job] Started job {job_id} with Celery task {task_id}"
    )
    return task_id


def start_collection_batch_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_collection_batch_job

    task_id = _enqueue_with_trace_context(
        run_collection_batch_job,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(
        f"[start_collection_batch_job] Started job {job_id} with Celery task {task_id}"
    )
    return task_id


def start_delete_collection_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_delete_collection_job

    task_id = _enqueue_with_trace_context(
        run_delete_collection_job,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(
        f"[start_delete_collection_job] Started job {job_id} with Celery task {task_id}"
    )
    return task_id


def start_stt_batch_submission(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_stt_batch_submission

    task_id = _enqueue_with_trace_context(
        run_stt_batch_submission,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(
        f"[start_stt_batch_submission] Started job {job_id} with Celery task {task_id}"
    )
    return task_id


def start_stt_metric_computation(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_stt_metric_computation

    task_id = _enqueue_with_trace_context(
        run_stt_metric_computation,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(
        f"[start_stt_metric_computation] Started job {job_id} with Celery task {task_id}"
    )
    return task_id


def start_tts_batch_submission(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_tts_batch_submission

    task_id = _enqueue_with_trace_context(
        run_tts_batch_submission,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(
        f"[start_tts_batch_submission] Started job {job_id} with Celery task {task_id}"
    )
    return task_id


def start_tts_result_processing(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_tts_result_processing

    task_id = _enqueue_with_trace_context(
        run_tts_result_processing,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )
    logger.info(
        f"[start_tts_result_processing] Started job {job_id} with Celery task {task_id}"
    )
    return task_id


def start_fast_evaluation(eval_run_id: int, trace_id: str = "N/A") -> str:
    """Enqueue the run_evaluation_fast orchestrator task for one EvaluationRun."""
    from app.celery.tasks.evaluation_fast import run_evaluation_fast

    task_id = _enqueue_with_trace_context(
        run_evaluation_fast,
        eval_run_id=eval_run_id,
        trace_id=trace_id,
    )
    logger.info(
        f"[start_fast_evaluation] Enqueued fast eval | "
        f"eval_run_id={eval_run_id} | task_id={task_id}"
    )
    return task_id


def get_task_status(task_id: str) -> Dict[str, Any]:
    result = AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result,
        "info": result.info,
    }


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    try:
        celery_app.control.revoke(task_id, terminate=terminate)
        logger.info(f"[revoke_task] Revoked task {task_id}")
        return True
    except Exception as e:
        logger.error(f"[revoke_task] Failed to revoke task {task_id}: {e}")
        return False


def gevent_timeout(
    seconds: float | None, task_name: str | None = None
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = task_name or func.__name__
            timeout = Timeout(seconds)
            timeout.start()
            try:
                return func(*args, **kwargs)
            except Timeout as err:
                if err is not timeout:
                    raise
                logger.error(f"[{name}] Timed out after {seconds}s")
                raise
            finally:
                timeout.cancel()

        return wrapper  # type: ignore[return-value]

    return decorator


# In gevent mode, Celery's soft and hard time limits fire during task cleanup,
# producing a misleading "Hard time limit exceeded" log. The task has already
# completed at this point (Pool POST fires first). This is a known gevent/Celery
# interaction and is harmless.
