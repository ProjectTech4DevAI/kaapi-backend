"""
Utility functions for easy Celery integration across the application.
Business logic modules can use these functions without knowing Celery internals.
"""
import logging
from typing import Any, Dict

from celery.result import AsyncResult

from app.celery.celery_app import celery_app

logger = logging.getLogger(__name__)


def start_llm_job(project_id: int, job_id: str, trace_id: str = "N/A", **kwargs) -> str:
    from app.celery.tasks.job_execution import run_llm_job

    task = run_llm_job.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(f"[start_llm_job] Started job {job_id} with Celery task {task.id}")
    return task.id


def start_llm_chain_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_llm_chain_job

    task = run_llm_chain_job.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(
        f"[start_llm_chain_job] Started job {job_id} with Celery task {task.id}"
    )
    return task.id


def start_response_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_response_job

    task = run_response_job.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(f"[start_response_job] Started job {job_id} with Celery task {task.id}")
    return task.id


def start_doctransform_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_doctransform_job

    task = run_doctransform_job.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(
        f"[start_doctransform_job] Started job {job_id} with Celery task {task.id}"
    )
    return task.id


def start_create_collection_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_create_collection_job

    task = run_create_collection_job.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(
        f"[start_create_collection_job] Started job {job_id} with Celery task {task.id}"
    )
    return task.id


def start_delete_collection_job(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_delete_collection_job

    task = run_delete_collection_job.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(
        f"[start_delete_collection_job] Started job {job_id} with Celery task {task.id}"
    )
    return task.id


def start_stt_batch_submission(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_stt_batch_submission

    task = run_stt_batch_submission.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(
        f"[start_stt_batch_submission] Started job {job_id} with Celery task {task.id}"
    )
    return task.id


def start_stt_metric_computation(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_stt_metric_computation

    task = run_stt_metric_computation.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(
        f"[start_stt_metric_computation] Started job {job_id} with Celery task {task.id}"
    )
    return task.id


def start_tts_batch_submission(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_tts_batch_submission

    task = run_tts_batch_submission.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(
        f"[start_tts_batch_submission] Started job {job_id} with Celery task {task.id}"
    )
    return task.id


def start_tts_result_processing(
    project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    from app.celery.tasks.job_execution import run_tts_result_processing

    task = run_tts_result_processing.delay(
        project_id=project_id, job_id=job_id, trace_id=trace_id, **kwargs
    )
    logger.info(
        f"[start_tts_result_processing] Started job {job_id} with Celery task {task.id}"
    )
    return task.id


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
