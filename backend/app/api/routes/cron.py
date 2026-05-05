import logging

import sentry_sdk
from sentry_sdk.types import MonitorConfig

from app.api.permissions import Permission, require_permission
from fastapi import APIRouter, Depends

from app.api.deps import SessionDep
from app.core.config import settings
from app.crud.evaluations import process_all_pending_evaluations_sync

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


@router.get(
    "/cron/evaluations",
    include_in_schema=False,
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
@sentry_sdk.monitor(
    monitor_slug="evaluation-cron-job",
    monitor_config=EVALUATION_CRON_MONITOR_CONFIG,
)
def evaluation_cron_job(
    session: SessionDep,
) -> dict:
    """
    Cron job endpoint for periodic evaluation tasks.

    This endpoint:
    1. Fetches all evaluation runs with status='processing'
    2. Groups them by project_id
    3. Processes each project with its OpenAI/Langfuse clients
    4. Returns aggregated results

    Hidden from Swagger documentation.
    Requires authentication via FIRST_SUPERUSER credentials.
    """
    logger.info("[evaluation_cron_job] Cron job invoked")

    try:
        result = process_all_pending_evaluations_sync(session=session)

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
