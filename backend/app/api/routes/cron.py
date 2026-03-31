import logging

import sentry_sdk
from sentry_sdk.types import MonitorConfig

from app.api.permissions import Permission, require_permission
from fastapi import APIRouter, Depends

from app.api.deps import SessionDep
from app.core.config import settings
from app.crud.evaluations import process_all_pending_evaluations

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

        # Also poll assessment evaluations (must await in the same event loop
        # so that SSE publish reaches subscribers via the shared broker).
        try:
            from app.assessment.cron import (
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
            result["assessment_error"] = str(ae)

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
