"""
Celery worker management script.
"""
import logging
import multiprocessing
from celery.bin import worker
from app.celery.celery_app import celery_app
from app.core.config import settings

logger = logging.getLogger(__name__)


def start_worker(
    queues: str = "default,high_priority,low_priority,cron",
    concurrency: int = None,
    loglevel: str = "info",
):
    """
    Start Celery worker with specified configuration.

    Args:
        queues: Comma-separated list of queues to consume
        concurrency: Number of worker processes (defaults to settings or CPU count)
        loglevel: Logging level
    """
    if concurrency is None:
        concurrency = settings.CELERY_WORKER_CONCURRENCY or multiprocessing.cpu_count()

    logger.info(f"Starting Celery worker with {concurrency} processes")
    logger.info(f"Consuming queues: {queues}")

    # Start the worker
    worker_instance = worker.worker(app=celery_app)
    worker_instance.run(
        queues=queues.split(","),
        concurrency=concurrency,
        loglevel=loglevel,
        without_gossip=True,
        without_mingle=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Celery worker")
    parser.add_argument(
        "--queues",
        default="default,high_priority,low_priority,cron",
        help="Comma-separated list of queues to consume",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of worker processes (defaults to config or CPU count)",
    )
    parser.add_argument(
        "--loglevel",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    args = parser.parse_args()
    start_worker(args.queues, args.concurrency, args.loglevel)
