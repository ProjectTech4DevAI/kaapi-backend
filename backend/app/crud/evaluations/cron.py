"""
CRUD operations for evaluation cron jobs.

This module provides functions that can be invoked periodically to process
pending evaluations across all organizations.
"""

import asyncio
import logging
from typing import Any

from sqlmodel import Session

from app.crud.evaluations.processing import poll_all_pending_evaluations

logger = logging.getLogger(__name__)


async def process_all_pending_evaluations(session: Session) -> dict[str, Any]:
    """
    Process all pending evaluations across all organizations.

    Delegates to poll_all_pending_evaluations which fetches all processing
    evaluation runs in a single query, groups by project, and processes them.

    Args:
        session: Database session

    Returns:
        Dict with aggregated results.
    """
    logger.info("[process_all_pending_evaluations] Starting evaluation processing")

    try:
        summary = await poll_all_pending_evaluations(session=session)

        logger.info(
            f"[process_all_pending_evaluations] Completed: "
            f"{summary['processed']} processed, {summary['failed']} failed, "
            f"{summary['still_processing']} still processing"
        )

        return {
            "status": "success",
            "total_processed": summary["processed"],
            "total_failed": summary["failed"],
            "total_still_processing": summary["still_processing"],
            "results": summary["details"],
        }

    except Exception as e:
        logger.error(
            f"[process_all_pending_evaluations] Fatal error: {e}",
            exc_info=True,
        )
        return {
            "status": "error",
            "total_processed": 0,
            "total_failed": 0,
            "total_still_processing": 0,
            "error": str(e),
            "results": [],
        }


def process_all_pending_evaluations_sync(session: Session) -> dict[str, Any]:
    """
    Synchronous wrapper for process_all_pending_evaluations.

    This function can be called from synchronous contexts (like FastAPI endpoints).

    Args:
        session: Database session

    Returns:
        Dict with aggregated results (same as process_all_pending_evaluations)
    """
    return asyncio.run(process_all_pending_evaluations(session=session))
