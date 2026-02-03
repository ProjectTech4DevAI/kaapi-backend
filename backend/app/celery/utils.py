"""
Utility functions for easy Celery integration across the application.
Business logic modules can use these functions without knowing Celery internals.
"""
import logging
from typing import Any, Dict, Optional
from celery.result import AsyncResult

from app.celery.celery_app import celery_app
from app.celery.tasks.job_execution import (
    execute_high_priority_task,
    execute_low_priority_task,
)

logger = logging.getLogger(__name__)


def start_high_priority_job(
    function_path: str, project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    """
    Start a high priority job using Celery.

    Args:
        function_path: Import path to the execute_job function (e.g., "app.services.doctransform.service.execute_job")
        project_id: ID of the project executing the job
        job_id: ID of the job (should already exist in database)
        trace_id: Trace/correlation ID to preserve context across Celery tasks
        **kwargs: Additional arguments to pass to the execute_job function

    Returns:
        Celery task ID (different from job_id)
    """
    task = execute_high_priority_task.delay(
        function_path=function_path,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )

    logger.info(f"Started high priority job {job_id} with Celery task {task.id}")
    return task.id


def start_low_priority_job(
    function_path: str, project_id: int, job_id: str, trace_id: str = "N/A", **kwargs
) -> str:
    """
    Start a low priority job using Celery.

    Args:
        function_path: Import path to the execute_job function (e.g., "app.services.doctransform.service.execute_job")
        project_id: ID of the project executing the job
        job_id: ID of the job (should already exist in database)
        trace_id: Trace/correlation ID to preserve context across Celery tasks
        **kwargs: Additional arguments to pass to the execute_job function

    Returns:
        Celery task ID (different from job_id)
    """
    task = execute_low_priority_task.delay(
        function_path=function_path,
        project_id=project_id,
        job_id=job_id,
        trace_id=trace_id,
        **kwargs,
    )

    logger.info(f"Started low priority job {job_id} with Celery task {task.id}")
    return task.id


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a Celery task.

    Args:
        task_id: Celery task ID

    Returns:
        Dictionary with task status information
    """
    result = AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result,
        "info": result.info,
    }


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """
    Revoke (cancel) a Celery task.

    Args:
        task_id: Celery task ID
        terminate: Whether to terminate the task if it's already running

    Returns:
        True if task was revoked successfully
    """
    try:
        celery_app.control.revoke(task_id, terminate=terminate)
        logger.info(f"Revoked task {task_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to revoke task {task_id}: {e}")
        return False
