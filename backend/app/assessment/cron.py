"""Cron processing functions for assessment evaluations."""

import logging
from typing import Any

from sqlmodel import Session, select

from app.assessment.crud import (
    get_assessment_runs_for_manager,
    recompute_assessment_status,
    update_assessment_run_status,
)
from app.assessment.events import assessment_event_broker
from app.assessment.models import Assessment, AssessmentRun
from app.assessment.processing import check_and_process_assessment
from app.utils import APIResponse

logger = logging.getLogger(__name__)


def _log_config_progress(result: dict[str, Any], run: AssessmentRun) -> None:
    """Emit explicit config-level logs for grouped assessment experiments."""
    action = result.get("action")
    if action not in {"processed", "failed"}:
        return

    logger.info(
        "[poll_all_pending_assessment_evaluations] "
        "Experiment config update | "
        f"experiment={run.run_name} | "
        f"assessment_id={run.assessment_id} | "
        f"run_id={run.id} | "
        f"config_id={run.config_id} | "
        f"config_version={run.config_version} | "
        f"action={action} | "
        f"status={result.get('current_status')} | "
        f"provider_status={result.get('provider_status')}"
    )


def _build_callback_payload(
    assessment: Assessment,
    run: AssessmentRun,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build minimal SSE payload for assessment invalidation."""
    return APIResponse.success_response(
        data={
            "type": "assessment.child_status_changed",
            "assessment_id": assessment.id,
            "assessment_status": assessment.status,
            "run": {
                "id": run.id,
                "config_id": str(run.config_id) if run.config_id else None,
                "config_version": run.config_version,
                "status": result.get("current_status"),
                "error": result.get("error"),
                "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            },
        }
    ).model_dump()


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
        runs = get_assessment_runs_for_manager(session=session, assessment=assessment)
        active_runs = [
            run for run in runs if run.status in {"processing", "in_progress"}
        ]

        if not active_runs:
            refreshed = recompute_assessment_status(
                session=session, assessment_id=assessment.id
            )
            logger.info(
                "[poll_all_pending_assessment_evaluations] "
                f"No active runs for assessment {assessment.id} | "
                f"recomputed status={refreshed.status} | "
                f"total_runs={refreshed.total_runs} | "
                f"completed={refreshed.completed_runs} | "
                f"failed={refreshed.failed_runs}"
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
                _log_config_progress(result, run)

                if result["action"] in {"processed", "failed"}:
                    refreshed_assessment = session.get(Assessment, assessment.id)
                    if refreshed_assessment:
                        assessment_event_broker.publish(
                            _build_callback_payload(
                                refreshed_assessment,
                                run,
                                result,
                            )
                        )

                if result["action"] == "processed":
                    processed += 1
                elif result["action"] == "failed":
                    failed += 1
                else:
                    still_processing += 1

            except Exception as e:
                logger.error(
                    "[poll_all_pending_assessment_evaluations] "
                    f"Failed run {run.id} | "
                    f"experiment={run.run_name} | "
                    f"assessment_id={run.assessment_id} | "
                    f"config_id={run.config_id} | "
                    f"config_version={run.config_version} | "
                    f"error={e}",
                    exc_info=True,
                )
                try:
                    update_assessment_run_status(
                        session=session,
                        run=run,
                        status="failed",
                        error_message="Processing failed. Check server logs for details.",
                    )
                    refreshed_assessment = recompute_assessment_status(
                        session=session, assessment_id=assessment.id
                    )
                    failure_result = {
                        "assessment_id": run.assessment_id,
                        "run_id": run.id,
                        "run_name": run.run_name,
                        "config_id": str(run.config_id) if run.config_id else None,
                        "config_version": run.config_version,
                        "action": "failed",
                        "error": "Processing failed",
                        "current_status": "failed",
                    }
                    all_results.append(failure_result)
                    assessment_event_broker.publish(
                        _build_callback_payload(
                            refreshed_assessment,
                            run,
                            failure_result,
                        )
                    )
                    failed += 1
                except Exception as cleanup_exc:
                    logger.error(
                        "[poll_all_pending_assessment_evaluations] "
                        f"Cleanup failed for run {run.id} | "
                        f"assessment_id={run.assessment_id} | "
                        f"run_name={run.run_name} | "
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
