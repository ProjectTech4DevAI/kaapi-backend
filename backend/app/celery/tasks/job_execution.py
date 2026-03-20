import logging
import importlib
import time
from celery import current_task
from asgi_correlation_id import correlation_id

from app.celery.celery_app import celery_app

logger = logging.getLogger(__name__)


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
    task_start = time.perf_counter()
    task_id = current_task.request.id

    t_start = time.perf_counter()
    correlation_id.set(trace_id)
    t_correlation = time.perf_counter()
    logger.info(
        f"[TIMING] Set correlation ID | duration={((t_correlation - t_start) * 1000):.2f}ms | job_id={job_id}"
    )

    try:
        # Dynamically import and resolve the function
        t_start = time.perf_counter()
        module_path, function_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        execute_function = getattr(module, function_name)
        t_import = time.perf_counter()

        logger.info(
            f"[TIMING] Dynamic import | duration={((t_import - t_start) * 1000):.2f}ms | module={module_path} | job_id={job_id}"
        )

        logger.info(
            f"Executing {priority} job {job_id} (task {task_id}) using function {function_path}"
        )

        # Execute the business logic function with standardized parameters
        t_start = time.perf_counter()
        result = execute_function(
            project_id=project_id,
            job_id=job_id,
            task_id=task_id,
            task_instance=task_instance,  # For progress updates, retries if needed
            **kwargs,
        )
        t_execute = time.perf_counter()

        task_total = time.perf_counter() - task_start

        # Calculate timings
        correlation_time = (t_correlation - task_start) * 1000
        import_time = (t_import - t_correlation) * 1000
        business_logic_time = (t_execute - t_import) * 1000
        total_task_time = task_total * 1000

        # Helper function for red highlighting if >1000ms
        def format_time(ms: float) -> str:
            if ms > 1000:
                return f"\033[91m{ms:>8.2f}ms ⚠️\033[0m"  # Red color
            return f"{ms:>8.2f}ms"

        logger.info(
            f"[TIMING] Business logic execution | duration={business_logic_time:.2f}ms | job_id={job_id}"
        )

        # Detailed Celery task breakdown
        logger.info(
            f"[TIMING] ═══════════════════════════════════════════════════════════"
        )
        logger.info(
            f"[TIMING] ═══ CELERY TASK EXECUTION BREAKDOWN (job_id={job_id}) ═══"
        )
        logger.info(
            f"[TIMING] ═══════════════════════════════════════════════════════════"
        )
        logger.info(f"[TIMING]   Pre-Worker Setup:")
        logger.info(
            f"[TIMING]     ├─ Correlation ID setup:      {format_time(correlation_time)}"
        )
        logger.info(
            f"[TIMING]     └─ Dynamic module import:     {format_time(import_time)}"
        )
        logger.info(
            f"[TIMING]   Business Logic (execute_job):  {format_time(business_logic_time)}"
        )
        logger.info(f"[TIMING]     └─ (see detailed breakdown above)")
        logger.info(
            f"[TIMING] ═══════════════════════════════════════════════════════════"
        )
        logger.info(
            f"[TIMING]   TOTAL CELERY TASK TIME:        {format_time(total_task_time)}"
        )
        logger.info(
            f"[TIMING] ═══════════════════════════════════════════════════════════"
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
