"""Cron processing functions for assessment evaluations."""

import logging
from typing import Any

from sqlmodel import Session, select

from app.assessment.crud import (
    compute_run_counts,
    get_assessment_runs_for_assessment,
    recompute_assessment_status,
    update_assessment_run_status,
)
from app.assessment.models import Assessment, AssessmentRun
from app.assessment.processing import (
    check_and_process_assessment,
    format_assessment_failure_message,
)

logger = logging.getLogger(__name__)


def _log_config_progress(
    result: dict[str, Any], run: AssessmentRun, assessment: Assessment
) -> None:
    """Emit explicit config-level logs for grouped assessment experiments."""
    action = result.get("action")
    if action not in {"processed", "failed"}:
        return

    logger.info(
        "[poll_all_pending_assessment_evaluations] "
        "Experiment config update | "
        f"experiment={assessment.experiment_name} | "
        f"assessment_id={run.assessment_id} | "
        f"run_id={run.id} | "
        f"config_id={run.config_id} | "
        f"config_version={run.config_version} | "
        f"action={action} | "
        f"status={result.get('current_status')} | "
        f"provider_status={result.get('provider_status')}"
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
        "[poll_all_pending_assessment_evaluations] "
        f"Found {len(pending_assessments)} active assessments"
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
            run for run in runs if run.status in {"processing", "in_progress"}
        ]

        if not active_runs:
            refreshed = recompute_assessment_status(
                session=session, assessment_id=assessment.id
            )
            counts = compute_run_counts(runs)
            logger.info(
                "[poll_all_pending_assessment_evaluations] "
                f"No active runs for assessment {assessment.id} | "
                f"recomputed status={refreshed.status} | "
                f"total_runs={counts.total} | "
                f"completed={counts.completed} | "
                f"failed={counts.failed}"
            )
            if refreshed.status in {"pending", "processing"}:
                still_processing += 1
            continue

        for run in active_runs:
            try:
                result = await check_and_process_assessment(
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
                error_msg = format_assessment_failure_message(e)
                logger.error(
                    "[poll_all_pending_assessment_evaluations] "
                    f"Failed run {run.id} | "
                    f"experiment={assessment.experiment_name} | "
                    f"assessment_id={run.assessment_id} | "
                    f"config_id={run.config_id} | "
                    f"config_version={run.config_version} | "
                    f"error={error_msg}",
                    exc_info=True,
                )
                try:
                    update_assessment_run_status(
                        session=session,
                        run=run,
                        status="failed",
                        error_message=error_msg,
                    )
                    recompute_assessment_status(
                        session=session, assessment_id=assessment.id
                    )
                    failure_result = {
                        "assessment_id": run.assessment_id,
                        "run_id": run.id,
                        "experiment_name": assessment.experiment_name,
                        "config_id": str(run.config_id) if run.config_id else None,
                        "config_version": run.config_version,
                        "action": "failed",
                        "error": error_msg,
                        "current_status": "failed",
                    }
                    all_results.append(failure_result)
                    failed += 1
                except Exception as cleanup_exc:
                    logger.error(
                        "[poll_all_pending_assessment_evaluations] "
                        f"Cleanup failed for run {run.id} | "
                        f"assessment_id={run.assessment_id} | "
                        f"experiment={assessment.experiment_name} | "
                        f"error={cleanup_exc}",
                        exc_info=True,
                    )
                    failed += 1

    logger.info(
        "[poll_all_pending_assessment_evaluations] Summary | "
        f"processed={processed} | failed={failed} | still_processing={still_processing}"
    )

    return {
        "total": len(pending_assessments),
        "processed": processed,
        "failed": failed,
        "still_processing": still_processing,
        "details": all_results,
    }
