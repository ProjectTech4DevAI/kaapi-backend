"""Cron processing functions for assessment evaluations."""

import logging
from typing import Any

from sqlmodel import Session, select

from app.crud.assessment import (
    compute_run_counts,
    get_assessment_runs_for_assessment,
    recompute_assessment_status,
)
from app.crud.assessment.processing import (
    format_assessment_failure_message,
    process_run_batches,
)
from app.models.assessment import Assessment, AssessmentRun, StageStatus

logger = logging.getLogger(__name__)


def _log_config_progress(
    result: dict[str, Any], run: AssessmentRun, assessment: Assessment
) -> None:
    """Emit explicit config-level logs for grouped assessment experiments."""
    action = result.get("action")
    if action not in {"processed", "failed"}:
        return

    logger.info(
        "[poll_all_pending_assessment_evaluations] Experiment config update | "
        "experiment=%s | assessment_id=%s | run_id=%s | config_id=%s | "
        "config_version=%s | action=%s | status=%s | provider_status=%s",
        assessment.experiment_name,
        run.assessment_id,
        run.id,
        run.config_id,
        run.config_version,
        action,
        result.get("current_status"),
        result.get("provider_status"),
    )


async def poll_all_pending_assessment_evaluations(
    session: Session,
) -> dict[str, Any]:
    """Poll all non-terminal parent assessments and their active child runs."""
    statement = select(Assessment).where(
        Assessment.status.in_(("pending", "processing")),
    )
    pending_assessments = list(session.exec(statement).all())

    if not pending_assessments:
        logger.info(
            "[poll_all_pending_assessment_evaluations] " "No active assessments found"
        )
        return {
            "total": 0,
            "processed": 0,
            "failed": 0,
            "still_processing": 0,
            "details": [],
        }

    logger.info(
        "[poll_all_pending_assessment_evaluations] Found %s active assessments",
        len(pending_assessments),
    )

    all_results: list[dict[str, Any]] = []
    processed = 0
    failed = 0
    still_processing = 0

    for assessment in pending_assessments:
        runs = get_assessment_runs_for_assessment(
            session=session, assessment_id=assessment.id
        )
        active_runs = [
            run for run in runs if run.stage_status == StageStatus.PROCESSING
        ]

        if not active_runs:
            refreshed = recompute_assessment_status(
                session=session, assessment_id=assessment.id
            )
            counts = compute_run_counts(runs)
            logger.info(
                "[poll_all_pending_assessment_evaluations] No active runs for assessment %s | "
                "recomputed status=%s | total_runs=%s | completed=%s | failed=%s",
                assessment.id,
                refreshed.status,
                counts.total,
                counts.completed,
                counts.failed,
            )
            if refreshed.status in {"pending", "processing"}:
                still_processing += 1
            continue

        for run in active_runs:
            try:
                result = await process_run_batches(
                    run=run,
                    session=session,
                )
                all_results.append(result)
                _log_config_progress(result, run, assessment)

                if result["action"] == "processed":
                    processed += 1
                elif result["action"] == "failed":
                    failed += 1
                else:
                    still_processing += 1

            except Exception as e:
                session.rollback()
                logger.warning(
                    "[poll_all_pending_assessment_evaluations] transient error polling "
                    "run %s (assessment %s), will retry: %s",
                    run.id,
                    run.assessment_id,
                    format_assessment_failure_message(e),
                )
                still_processing += 1

    logger.info(
        "[poll_all_pending_assessment_evaluations] Summary | processed=%s | failed=%s | still_processing=%s",
        processed,
        failed,
        still_processing,
    )

    return {
        "total": len(pending_assessments),
        "processed": processed,
        "failed": failed,
        "still_processing": still_processing,
        "details": all_results,
    }
