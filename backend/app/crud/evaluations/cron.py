"""
CRUD operations for evaluation cron jobs.

This module provides functions that can be invoked periodically to process
pending evaluations across all organizations.
"""

import asyncio
import logging
from typing import Any

from sqlmodel import Session, select

from app.crud.evaluations.processing import poll_all_pending_evaluations
from app.crud.stt_evaluations import poll_all_pending_stt_evaluations
from app.models import Organization

logger = logging.getLogger(__name__)


async def process_all_pending_evaluations(session: Session) -> dict[str, Any]:
    """
    Process all pending evaluations across all organizations.

    This function:
    1. Gets all organizations
    2. For each org, polls their pending evaluations
    3. Processes completed batches automatically
    4. Returns aggregated results

    This is the main function that should be called by the cron endpoint.

    Args:
        session: Database session

    Returns:
        Dict with aggregated results:
        {
            "status": "success",
            "organizations_processed": 3,
            "total_processed": 5,
            "total_failed": 1,
            "total_still_processing": 2,
            "results": [
                {
                    "org_id": 1,
                    "org_name": "Org 1",
                    "summary": {...}
                },
                ...
            ]
        }
    """
    logger.info("[process_all_pending_evaluations] Starting evaluation processing")

    try:
        # Get all organizations
        orgs = session.exec(select(Organization)).all()

        if not orgs:
            logger.info("[process_all_pending_evaluations] No organizations found")
            return {
                "status": "success",
                "organizations_processed": 0,
                "total_processed": 0,
                "total_failed": 0,
                "total_still_processing": 0,
                "message": "No organizations to process",
                "results": [],
            }

        logger.info(
            f"[process_all_pending_evaluations] Found {len(orgs)} organizations to process"
        )

        results = []
        total_processed = 0
        total_failed = 0
        total_still_processing = 0

        # Process each organization
        for org in orgs:
            try:
                logger.info(
                    f"[process_all_pending_evaluations] Processing org_id={org.id} ({org.name})"
                )

                # Poll all pending text evaluations for this org
                summary = await poll_all_pending_evaluations(
                    session=session, org_id=org.id
                )

                # Poll all pending STT evaluations for this org
                stt_summary = await poll_all_pending_stt_evaluations(
                    session=session, org_id=org.id
                )

                # Merge summaries
                combined_summary = {
                    "text": summary,
                    "stt": stt_summary,
                    "processed": summary.get("processed", 0)
                    + stt_summary.get("processed", 0),
                    "failed": summary.get("failed", 0) + stt_summary.get("failed", 0),
                    "still_processing": summary.get("still_processing", 0)
                    + stt_summary.get("still_processing", 0),
                }

                results.append(
                    {
                        "org_id": org.id,
                        "org_name": org.name,
                        "summary": combined_summary,
                    }
                )

                total_processed += combined_summary["processed"]
                total_failed += combined_summary["failed"]
                total_still_processing += combined_summary["still_processing"]

            except Exception as e:
                logger.error(
                    f"[process_all_pending_evaluations] Error processing org_id={org.id}: {e}",
                    exc_info=True,
                )
                session.rollback()
                results.append(
                    {"org_id": org.id, "org_name": org.name, "error": str(e)}
                )
                total_failed += 1

        logger.info(
            f"[process_all_pending_evaluations] Completed: "
            f"{total_processed} processed, {total_failed} failed, "
            f"{total_still_processing} still processing"
        )

        return {
            "status": "success",
            "organizations_processed": len(orgs),
            "total_processed": total_processed,
            "total_failed": total_failed,
            "total_still_processing": total_still_processing,
            "results": results,
        }

    except Exception as e:
        logger.error(
            f"[process_all_pending_evaluations] Fatal error: {e}",
            exc_info=True,
        )
        return {
            "status": "error",
            "organizations_processed": 0,
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
