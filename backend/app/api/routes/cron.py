import logging

from app.api.permissions import Permission, require_permission
from fastapi import APIRouter, Depends

from app.api.deps import SessionDep
from app.crud.evaluations import process_all_pending_evaluations

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Cron"])


@router.get(
    "/cron/evaluations",
    include_in_schema=False,
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
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
        return {
            "status": "error",
            "error": str(e),
            "total_processed": 0,
            "total_failed": 0,
            "total_still_processing": 0,
        }
