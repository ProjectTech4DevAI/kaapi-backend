import logging

from celery import Celery
from celery.signals import (
    worker_init,
    task_failure,
    task_postrun,
    task_prerun,
    worker_process_init,
)
from kombu import Exchange, Queue

from app.core.config import settings
from app.core.logger import configure_logging
from app.core.sentry_filters import before_send_transaction_filter

configure_logging(service_name="kaapi-celery")

logger = logging.getLogger(__name__)
_telemetry_initialized = False
_sentry_initialized = False
_flush_hook_registered = False


def _initialize_worker_observability() -> None:
    global _telemetry_initialized, _sentry_initialized, _flush_hook_registered

    if settings.SENTRY_DSN and not _sentry_initialized:
        import sentry_sdk
        from sentry_sdk.integrations.celery import CeleryIntegration
        from sentry_sdk.integrations.httpx import HttpxIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=str(settings.SENTRY_DSN),
            environment=settings.ENVIRONMENT,
            release=settings.API_VERSION,
            instrumenter="otel",
            traces_sample_rate=1.0,
            enable_logs=True,
            before_send_transaction=before_send_transaction_filter,
            integrations=[
                LoggingIntegration(
                    level=logging.INFO,
                    sentry_logs_level=logging.INFO,
                ),
                CeleryIntegration(
                    propagate_traces=False,
                    monitor_beat_tasks=False,
                ),
            ],
            disabled_integrations=[
                SqlalchemyIntegration(),
                HttpxIntegration(),
            ],
        )
        _sentry_initialized = True

    if not _telemetry_initialized:
        from app.core.telemetry import flush_telemetry, setup_telemetry

        setup_telemetry(service_name="kaapi-celery")

        if not _flush_hook_registered:

            @task_postrun.connect(weak=False)
            def _flush_otel_after_task(**_: object) -> None:
                flush_telemetry()

            _flush_hook_registered = True

        _telemetry_initialized = True


@task_prerun.connect
def log_pool_status(task: "celery.Task", **_: object) -> None:  # type: ignore[name-defined]
    """Log DB connection pool state before each task to detect connection leaks.

    If checked_out equals pool size right when a task starts, connections
    are being held across tasks (likely across LLM API calls) and leaking.
    """
    from app.core.db import engine
    from sqlalchemy.pool import QueuePool

    pool = engine.pool
    if isinstance(pool, QueuePool):
        logger.info(
            f"[pool] task={task.name} checked_out={pool.checkedout()} "
            f"size={pool.size()} overflow={pool.overflow()}"
        )


@task_postrun.connect
def log_pool_status_post(task: "celery.Task", **_: object) -> None:  # type: ignore[name-defined]
    """Log DB connection pool state after each task completes (success or failure).

    Compare with task_prerun log — if checked_out is the same or higher after
    the task, connections were not returned and are leaking.
    """
    from app.core.db import engine
    from sqlalchemy.pool import QueuePool

    pool = engine.pool
    if isinstance(pool, QueuePool):
        logger.info(
            f"[pool] POST task={task.name} checked_out={pool.checkedout()} "
            f"size={pool.size()} overflow={pool.overflow()}"
        )


@task_failure.connect
def log_pool_status_failure(
    task_id: str, exception: Exception, sender: "celery.Task", **_: object  # type: ignore[name-defined]
) -> None:
    """Log DB connection pool state on task failure.

    Failures are the most likely path for sessions to leak — exceptions can
    bypass session cleanup if not guarded by a context manager.
    """
    from app.core.db import engine
    from sqlalchemy.pool import QueuePool

    pool = engine.pool
    if isinstance(pool, QueuePool):
        logger.warning(
            f"[pool] FAILED task={sender.name} task_id={task_id} "
            f"exc={type(exception).__name__} "
            f"checked_out={pool.checkedout()} "
            f"size={pool.size()} overflow={pool.overflow()}"
        )


@worker_process_init.connect
def initialize_worker_process(**_) -> None:
    """Initialize each forked Celery worker process.

    - Initialize Sentry so task transactions, errors, and logs ship to Sentry.
    - Set up OpenTelemetry so Celery tasks emit spans bridged into Sentry.
    - Pre-import LLM modules so first-call latency doesn't pay a cold-import cost.
    """
    _initialize_worker_observability()

    import app.services.llm.jobs  # noqa: F401


@worker_init.connect
def initialize_worker(**_) -> None:
    """Initialize worker observability for non-prefork pools (e.g. gevent)."""
    _initialize_worker_observability()


# Create Celery instance
celery_app = Celery(
    "ai_platform",
    broker=settings.RABBITMQ_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.celery.tasks.job_execution",
    ],
)

# Define exchanges and queues with priority
default_exchange = Exchange("default", type="direct")

# Celery configuration using environment variables
celery_app.conf.update(
    # Queue configuration with priority support
    task_queues=(
        Queue(
            "high_priority",
            exchange=default_exchange,
            routing_key="high",
            queue_arguments={"x-max-priority": 10},
        ),
        Queue(
            "low_priority",
            exchange=default_exchange,
            routing_key="low",
            queue_arguments={"x-max-priority": 1},
        ),
        Queue("cron", exchange=default_exchange, routing_key="cron"),
        Queue("default", exchange=default_exchange, routing_key="default"),
    ),
    # Task routing — queue is set per-task via @celery_app.task(queue=...).
    # Only cron tasks need an explicit override here.
    task_routes={
        "app.celery.tasks.*_cron_*": {"queue": "cron"},
    },
    task_default_queue="default",
    # Enable priority support
    task_inherit_parent_priority=True,
    worker_prefetch_multiplier=settings.CELERY_WORKER_PREFETCH_MULTIPLIER,
    # Worker configuration from environment
    worker_concurrency=settings.COMPUTED_CELERY_WORKER_CONCURRENCY,
    worker_max_tasks_per_child=settings.CELERY_WORKER_MAX_TASKS_PER_CHILD,
    worker_max_memory_per_child=settings.CELERY_WORKER_MAX_MEMORY_PER_CHILD,
    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,
    # Task execution from environment
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
    task_acks_late=True,
    # Retry configuration from environment
    task_default_retry_delay=settings.CELERY_TASK_DEFAULT_RETRY_DELAY,
    task_max_retries=settings.CELERY_TASK_MAX_RETRIES,
    # Task configuration from environment
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=settings.CELERY_ENABLE_UTC,
    task_track_started=True,
    task_always_eager=False,
    # Result backend settings from environment
    result_expires=settings.CELERY_RESULT_EXPIRES,
    result_compression="gzip",
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Connection settings from environment
    broker_connection_retry_on_startup=True,
    broker_pool_limit=settings.CELERY_BROKER_POOL_LIMIT,
)
