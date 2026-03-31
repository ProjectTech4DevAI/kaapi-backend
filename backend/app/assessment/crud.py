"""Assessment CRUD — thin wrappers around EvaluationRun for type='assessment'."""

import logging
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.core.util import now
from app.assessment.models import Assessment
from app.models.evaluation import EvaluationRun

logger = logging.getLogger(__name__)

ASSESSMENT_TYPE = "assessment"


def create_assessment(
    session: Session,
    experiment_name: str,
    dataset_id: int,
    dataset_name: str,
    organization_id: int,
    project_id: int,
    total_runs: int,
) -> Assessment:
    """Create a parent assessment manager row."""
    assessment = Assessment(
        experiment_name=experiment_name,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        status="pending",
        total_runs=total_runs,
        pending_runs=total_runs,
        processing_runs=0,
        completed_runs=0,
        failed_runs=0,
        run_stats=[],
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )

    session.add(assessment)
    try:
        session.commit()
        session.refresh(assessment)
    except Exception as e:
        session.rollback()
        logger.error(f"[create_assessment] Failed: {e}", exc_info=True)
        raise

    logger.info(
        f"[create_assessment] Created assessment id={assessment.id} | "
        f"experiment={experiment_name} | total_runs={total_runs}"
    )
    return assessment


def get_assessment_runs_for_manager(
    session: Session,
    assessment: Assessment,
) -> list[EvaluationRun]:
    """List child evaluation runs for a parent assessment row."""
    statement = (
        select(EvaluationRun)
        .where(EvaluationRun.assessment_id == assessment.id)
        .where(EvaluationRun.type == ASSESSMENT_TYPE)
        .order_by(EvaluationRun.inserted_at.desc())
    )
    return list(session.exec(statement).all())


def create_assessment_run(
    session: Session,
    run_name: str,
    dataset_name: str,
    dataset_id: int,
    assessment_id: int | None,
    config_id: UUID,
    config_version: int,
    organization_id: int,
    project_id: int,
    assessment_config: dict[str, Any] | None = None,
) -> EvaluationRun:
    """Create an assessment evaluation run record.

    Re-uses EvaluationRun with type='assessment' and stores assessment-specific
    config in the score JSONB field under 'assessment_config'.
    """
    eval_run = EvaluationRun(
        run_name=run_name,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        assessment_id=assessment_id,
        type=ASSESSMENT_TYPE,
        config_id=config_id,
        config_version=config_version,
        status="pending",
        score={"assessment_config": assessment_config} if assessment_config else None,
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )

    session.add(eval_run)
    try:
        session.commit()
        session.refresh(eval_run)
    except Exception as e:
        session.rollback()
        logger.error(f"[create_assessment_run] Failed: {e}", exc_info=True)
        raise

    logger.info(
        f"[create_assessment_run] Created run id={eval_run.id} | "
        f"name={run_name} | config_id={config_id} v{config_version}"
    )
    return eval_run


def list_assessments(
    session: Session,
    organization_id: int,
    project_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[Assessment]:
    """List parent assessment manager rows."""
    statement = (
        select(Assessment)
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
        .order_by(Assessment.inserted_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.exec(statement).all())


def get_assessment_by_id(
    session: Session,
    assessment_id: int,
    organization_id: int,
    project_id: int,
) -> Assessment | None:
    """Get a specific parent assessment manager row."""
    statement = (
        select(Assessment)
        .where(Assessment.id == assessment_id)
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
    )
    return session.exec(statement).first()


def list_assessment_runs(
    session: Session,
    organization_id: int,
    project_id: int,
    assessment_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[EvaluationRun]:
    """List child assessment evaluation runs by traversing from assessments."""
    if assessment_id is not None:
        assessment = get_assessment_by_id(
            session=session,
            assessment_id=assessment_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if not assessment:
            return []
        runs = get_assessment_runs_for_manager(session=session, assessment=assessment)
        return runs[offset : offset + limit]

    assessments = list_assessments(
        session=session,
        organization_id=organization_id,
        project_id=project_id,
        limit=limit,
        offset=offset,
    )
    runs: list[EvaluationRun] = []
    for assessment in assessments:
        runs.extend(
            get_assessment_runs_for_manager(session=session, assessment=assessment)
        )
    return runs


def get_assessment_run_by_id(
    session: Session,
    run_id: int,
    organization_id: int,
    project_id: int,
) -> EvaluationRun | None:
    """Get a specific assessment evaluation run by ID."""
    statement = (
        select(EvaluationRun)
        .where(EvaluationRun.id == run_id)
        .where(EvaluationRun.organization_id == organization_id)
        .where(EvaluationRun.project_id == project_id)
        .where(EvaluationRun.type == ASSESSMENT_TYPE)
    )
    return session.exec(statement).first()


def _determine_assessment_status(
    total_runs: int,
    pending_runs: int,
    processing_runs: int,
    completed_runs: int,
    failed_runs: int,
) -> str:
    """Compute parent assessment status from child evaluation runs."""
    if total_runs == 0:
        return "pending"
    if completed_runs == total_runs:
        return "completed"
    if failed_runs == total_runs:
        return "failed"
    if (
        completed_runs > 0
        and failed_runs > 0
        and pending_runs == 0
        and processing_runs == 0
    ):
        return "completed_with_errors"
    if pending_runs > 0 and pending_runs == total_runs:
        return "pending"
    return "processing"


def recompute_assessment_status(
    session: Session,
    assessment_id: int,
) -> Assessment:
    """Recompute cached parent assessment counters from child runs."""
    assessment = session.get(Assessment, assessment_id)
    if not assessment:
        raise ValueError(f"Assessment {assessment_id} not found")

    statement = (
        select(EvaluationRun)
        .where(EvaluationRun.assessment_id == assessment_id)
        .where(EvaluationRun.type == ASSESSMENT_TYPE)
        .order_by(EvaluationRun.id.asc())
    )
    runs = list(session.exec(statement).all())

    pending_runs = sum(1 for run in runs if run.status == "pending")
    processing_runs = sum(
        1 for run in runs if run.status in {"processing", "in_progress"}
    )
    completed_runs = sum(1 for run in runs if run.status == "completed")
    failed_runs = sum(1 for run in runs if run.status == "failed")
    total_runs = len(runs)

    assessment.total_runs = total_runs
    assessment.pending_runs = pending_runs
    assessment.processing_runs = processing_runs
    assessment.completed_runs = completed_runs
    assessment.failed_runs = failed_runs
    assessment.status = _determine_assessment_status(
        total_runs=total_runs,
        pending_runs=pending_runs,
        processing_runs=processing_runs,
        completed_runs=completed_runs,
        failed_runs=failed_runs,
    )
    assessment.error_message = (
        f"{failed_runs} of {total_runs} evaluation run(s) failed"
        if failed_runs > 0
        else None
    )
    assessment.run_stats = [
        {
            "run_id": run.id,
            "config_id": str(run.config_id) if run.config_id else None,
            "config_version": run.config_version,
            "status": run.status,
            "total_items": run.total_items,
            "error_message": run.error_message,
            "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        }
        for run in runs
    ]
    assessment.updated_at = now()

    session.add(assessment)
    try:
        session.commit()
        session.refresh(assessment)
    except Exception as e:
        session.rollback()
        logger.error(f"[recompute_assessment_status] Failed: {e}", exc_info=True)
        raise

    return assessment


def update_assessment_run_status(
    session: Session,
    eval_run: EvaluationRun,
    status: str,
    error_message: str | None = None,
    batch_job_id: int | None = None,
    total_items: int | None = None,
    object_store_url: str | None = None,
) -> EvaluationRun:
    """Update an assessment run's status and optional fields."""
    eval_run.status = status
    eval_run.updated_at = now()

    if error_message is not None:
        eval_run.error_message = error_message
    if batch_job_id is not None:
        eval_run.batch_job_id = batch_job_id
    if total_items is not None:
        eval_run.total_items = total_items
    if object_store_url is not None:
        eval_run.object_store_url = object_store_url

    session.add(eval_run)
    try:
        session.commit()
        session.refresh(eval_run)
    except Exception as e:
        session.rollback()
        logger.error(f"[update_assessment_run_status] Failed: {e}", exc_info=True)
        raise

    return eval_run
