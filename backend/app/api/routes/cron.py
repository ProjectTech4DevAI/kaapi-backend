import logging
from typing import Any

import sentry_sdk
from fastapi import APIRouter, Depends
from sentry_sdk.types import MonitorConfig

from app.api.deps import SessionDep
from app.api.permissions import Permission, require_permission
from app.core.config import settings
from app.crud.evaluations import process_all_pending_evaluations
from app.services.job_monitoring import monitor_pending_jobs
from app.crud.stats import get_daily_stats
from app.services.stats import format_sections, post_to_discord

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Cron"])

EVALUATION_CRON_MONITOR_CONFIG: MonitorConfig = {
    # Expected cadence: a check-in every CRON_INTERVAL_MINUTES minutes.
    "schedule": {
        "type": "interval",
        "value": settings.CRON_INTERVAL_MINUTES,
        "unit": "minute",
    },
    # Timezone for the schedule (only affects crontab-style schedules).
    "timezone": "UTC",
    # Grace period (minutes) before a late check-in is marked as missed.
    "checkin_margin": 2,
    # Max runtime (minutes) before an in-progress run is marked as timed out.
    "max_runtime": 2 * settings.CRON_INTERVAL_MINUTES,
    # Consecutive failures/missed/timeouts required to open a Sentry issue.
    "failure_issue_threshold": 2,
    # Consecutive successful check-ins required to auto-resolve the issue.
    "recovery_threshold": 1,
}

DAILY_STATS_CRON_MONITOR_CONFIG: MonitorConfig = {
    "schedule": {"type": "crontab", "value": "0 9 * * *"},
    "timezone": "UTC",
    "checkin_margin": 5,
    "max_runtime": 10,
    "failure_issue_threshold": 1,
    "recovery_threshold": 1,
}


PENDING_JOBS_CRON_MONITOR_CONFIG: MonitorConfig = {
    "schedule": {
        "type": "interval",
        "value": settings.PENDING_JOB_MONITOR_INTERVAL_MINUTES,
        "unit": "minute",
    },
    "timezone": "Asia/Kolkata",
    "checkin_margin": 2,
    "max_runtime": 2 * settings.PENDING_JOB_MONITOR_INTERVAL_MINUTES,
    "failure_issue_threshold": 2,
    "recovery_threshold": 1,
}


@router.get(
    "/cron/evaluations",
    include_in_schema=False,
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
@sentry_sdk.monitor(
    monitor_slug="evaluation-cron-job",
    monitor_config=EVALUATION_CRON_MONITOR_CONFIG,
)
async def evaluation_cron_job(
    session: SessionDep,
) -> dict:
    """
    Cron job endpoint for periodic evaluation tasks.

    This endpoint:
    1. Fetches all evaluation runs with status='processing'
    2. Groups them by project_id
    3. Processes each project with its OpenAI/Langfuse clients
    4. Also polls pending assessment evaluations
    5. Returns aggregated results

    Hidden from Swagger documentation.
    Requires authentication via FIRST_SUPERUSER credentials.
    """
    logger.info("[evaluation_cron_job] Cron job invoked")

    try:
        # Process all pending evaluations across all organizations
        result = await process_all_pending_evaluations(session=session)

        try:
            from app.crud.assessment.cron import (
                poll_all_pending_assessment_evaluations,
            )

            assessment_result = await poll_all_pending_assessment_evaluations(
                session=session
            )

            # Merge assessment results into the main result
            result["assessment"] = assessment_result
            result["total_processed"] = result.get(
                "total_processed", 0
            ) + assessment_result.get("processed", 0)
            result["total_failed"] = result.get(
                "total_failed", 0
            ) + assessment_result.get("failed", 0)
            result["total_still_processing"] = result.get(
                "total_still_processing", 0
            ) + assessment_result.get("still_processing", 0)
        except Exception as ae:
            logger.error(
                f"[evaluation_cron_job] Assessment polling failed: {ae}",
                exc_info=True,
            )
            result["assessment_error"] = "Assessment polling failed"

        logger.info(
            f"[evaluation_cron_job] Completed: "
            f"processed={result.get('total_processed', 0)}, "
            f"failed={result.get('total_failed', 0)}, "
            f"still_processing={result.get('total_still_processing', 0)}"
        )

        return result

    except Exception as e:
        logger.error(
            f"[evaluation_cron_job] Error executing cron job: {e}",
            exc_info=True,
        )
        sentry_sdk.capture_exception(e)
        raise


@router.get(
    "/cron/daily-stats",
    include_in_schema=False,
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
@sentry_sdk.monitor(
    monitor_slug="daily-stats-cron-job",
    monitor_config=DAILY_STATS_CRON_MONITOR_CONFIG,
)
def daily_stats_cron_job(session: SessionDep) -> dict[str, Any]:
    try:
        stats = get_daily_stats(session=session)
        post_to_discord(format_sections(stats))
        return stats
    except Exception as e:
        logger.error(
            f"[daily_stats_cron_job] Error executing cron job: {e}",
            exc_info=True,
        )
        sentry_sdk.capture_exception(e)
        raise


@router.get(
    "/cron/pending-jobs",
    include_in_schema=False,
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
@sentry_sdk.monitor(
    monitor_slug="pending-jobs-monitor",
    monitor_config=PENDING_JOBS_CRON_MONITOR_CONFIG,
)
async def pending_jobs_cron_job(
    session: SessionDep,
) -> dict:
    """
    Cron endpoint for monitoring stale PENDING background jobs.

    Hidden from Swagger documentation.
    Requires authentication via FIRST_SUPERUSER credentials.
    """
    logger.info("[pending_jobs_cron_job] Cron job invoked")

    try:
        result = monitor_pending_jobs(session=session)
        logger.info(
            "[pending_jobs_cron_job] Completed: stale_pending=%s",
            result.get("total_stale_pending", 0),
        )
        return result
    except Exception as e:
        logger.error(
            f"[pending_jobs_cron_job] Error executing cron job: {e}",
            exc_info=True,
        )
        sentry_sdk.capture_exception(e)
        raise
