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
from app.assessment.processing import check_and_process_assessment
from app.assessment.models import Assessment
from app.models.evaluation import EvaluationRun
from app.utils import APIResponse

logger = logging.getLogger(__name__)


def _log_config_progress(result: dict[str, Any], eval_run: EvaluationRun) -> None:
    """Emit explicit config-level logs for grouped assessment experiments."""
    action = result.get("action")
    if action not in {"processed", "failed"}:
        return

    logger.info(
        "[poll_all_pending_assessment_evaluations] "
        "Experiment config update | "
        f"experiment={eval_run.run_name} | "
        f"assessment_id={eval_run.assessment_id} | "
        f"run_id={eval_run.id} | "
        f"config_id={eval_run.config_id} | "
        f"config_version={eval_run.config_version} | "
        f"action={action} | "
        f"status={result.get('current_status')} | "
        f"provider_status={result.get('provider_status')}"
    )


def _build_callback_payload(
    assessment: Assessment,
    eval_run: EvaluationRun,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build minimal SSE payload for assessment invalidation."""
    return APIResponse.success_response(
        data={
            "type": "assessment.child_status_changed",
            "assessment_id": assessment.id,
            "assessment_status": assessment.status,
            "run": {
                "id": eval_run.id,
                "config_id": str(eval_run.config_id) if eval_run.config_id else None,
                "config_version": eval_run.config_version,
                "status": result.get("current_status"),
                "error": result.get("error"),
                "updated_at": eval_run.updated_at.isoformat()
                if eval_run.updated_at
                else None,
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
            if refreshed.status in {"pending", "processing"}:
                still_processing += 1
            continue

        for eval_run in active_runs:
            try:
                result = await check_and_process_assessment(
                    eval_run=eval_run,
                    session=session,
                )
                all_results.append(result)
                _log_config_progress(result, eval_run)

                if result["action"] in {"processed", "failed"}:
                    refreshed_assessment = session.get(Assessment, assessment.id)
                    if refreshed_assessment:
                        assessment_event_broker.publish(
                            _build_callback_payload(
                                refreshed_assessment,
                                eval_run,
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
                    f"Failed run {eval_run.id} | "
                    f"experiment={eval_run.run_name} | "
                    f"assessment_id={eval_run.assessment_id} | "
                    f"config_id={eval_run.config_id} | "
                    f"config_version={eval_run.config_version} | "
                    f"error={e}",
                    exc_info=True,
                )
                update_assessment_run_status(
                    session=session,
                    eval_run=eval_run,
                    status="failed",
                    error_message=f"Poll failed: {str(e)}",
                )
                refreshed_assessment = recompute_assessment_status(
                    session=session, assessment_id=assessment.id
                )
                failure_result = {
                    "assessment_id": eval_run.assessment_id,
                    "run_id": eval_run.id,
                    "run_name": eval_run.run_name,
                    "config_id": str(eval_run.config_id)
                    if eval_run.config_id
                    else None,
                    "config_version": eval_run.config_version,
                    "action": "failed",
                    "error": str(e),
                    "current_status": "failed",
                }
                all_results.append(failure_result)
                assessment_event_broker.publish(
                    _build_callback_payload(
                        refreshed_assessment,
                        eval_run,
                        failure_result,
                    )
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
