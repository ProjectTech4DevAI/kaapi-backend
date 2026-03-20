import logging

from celery import Celery
from kombu import Exchange, Queue

from app.core.config import settings

logger = logging.getLogger(__name__)


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
    # Task routing
    task_routes={
        "app.celery.tasks.job_execution.execute_high_priority_task": {
            "queue": "high_priority",
            "priority": 9,
        },
        "app.celery.tasks.job_execution.execute_low_priority_task": {
            "queue": "low_priority",
            "priority": 1,
        },
        "app.celery.tasks.*_cron_*": {"queue": "cron"},
        "app.celery.tasks.*": {"queue": "default"},
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

# Auto-discover tasks
celery_app.autodiscover_tasks()
