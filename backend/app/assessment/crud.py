"""Assessment CRUD — operations for Assessment and AssessmentRun tables."""

import logging
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.assessment.models import (
    Assessment,
    AssessmentRun,
    AssessmentRunCounts,
    AssessmentRunStat,
)
from app.core.util import now

logger = logging.getLogger(__name__)


def create_assessment(
    session: Session,
    experiment_name: str,
    dataset_id: int,
    organization_id: int,
    project_id: int,
) -> Assessment:
    """Create a parent assessment row."""
    assessment = Assessment(
        experiment_name=experiment_name,
        dataset_id=dataset_id,
        status="pending",
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
        f"experiment={experiment_name}"
    )
    return assessment


def get_assessment_by_id(
    session: Session,
    assessment_id: int,
    organization_id: int,
    project_id: int,
) -> Assessment | None:
    """Get a specific parent assessment row."""
    statement = (
        select(Assessment)
        .where(Assessment.id == assessment_id)
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
    )
    return session.exec(statement).first()


def list_assessments(
    session: Session,
    organization_id: int,
    project_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[Assessment]:
    """List parent assessment rows."""
    statement = (
        select(Assessment)
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
        .order_by(Assessment.inserted_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.exec(statement).all())


def create_assessment_run(
    session: Session,
    assessment_id: int,
    config_id: UUID,
    config_version: int,
    assessment_input: dict[str, Any],
) -> AssessmentRun:
    """Create an assessment run record under a parent assessment."""
    run = AssessmentRun(
        assessment_id=assessment_id,
        config_id=config_id,
        config_version=config_version,
        status="pending",
        total_items=0,
        input=assessment_input,
        inserted_at=now(),
        updated_at=now(),
    )

    session.add(run)
    try:
        session.commit()
        session.refresh(run)
    except Exception as e:
        session.rollback()
        logger.error(f"[create_assessment_run] Failed: {e}", exc_info=True)
        raise

    logger.info(
        f"[create_assessment_run] Created run id={run.id} | "
        f"assessment_id={assessment_id} | "
        f"config_id={config_id} v{config_version}"
    )
    return run


def get_assessment_run_by_id(
    session: Session,
    run_id: int,
    organization_id: int,
    project_id: int,
) -> AssessmentRun | None:
    """Get a specific assessment run by ID, scoped via parent organization/project."""
    statement = (
        select(AssessmentRun)
        .join(Assessment, Assessment.id == AssessmentRun.assessment_id)
        .where(AssessmentRun.id == run_id)
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
    )
    return session.exec(statement).first()


def get_assessment_runs_for_assessment(
    session: Session,
    assessment_id: int,
) -> list[AssessmentRun]:
    """List child runs for a parent assessment, ordered by id."""
    statement = (
        select(AssessmentRun)
        .where(AssessmentRun.assessment_id == assessment_id)
        .order_by(AssessmentRun.id.asc())
    )
    return list(session.exec(statement).all())


def list_assessment_runs(
    session: Session,
    organization_id: int,
    project_id: int,
    assessment_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[AssessmentRun]:
    """List assessment runs, optionally filtered by assessment_id."""
    statement = (
        select(AssessmentRun)
        .join(Assessment, Assessment.id == AssessmentRun.assessment_id)
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
    )
    if assessment_id is not None:
        statement = statement.where(AssessmentRun.assessment_id == assessment_id)

    statement = (
        statement.order_by(AssessmentRun.inserted_at.desc()).limit(limit).offset(offset)
    )
    return list(session.exec(statement).all())


def update_assessment_run_status(
    session: Session,
    run: AssessmentRun,
    status: str,
    error_message: str | None = None,
    batch_job_id: int | None = None,
    total_items: int | None = None,
    object_store_url: str | None = None,
) -> AssessmentRun:
    """Update an assessment run's status and optional fields."""
    run.status = status
    run.updated_at = now()

    if error_message is not None:
        run.error_message = error_message
    if batch_job_id is not None:
        run.batch_job_id = batch_job_id
    if total_items is not None:
        run.total_items = total_items
    if object_store_url is not None:
        run.object_store_url = object_store_url

    session.add(run)
    try:
        session.commit()
        session.refresh(run)
    except Exception as e:
        session.rollback()
        logger.error(f"[update_assessment_run_status] Failed: {e}", exc_info=True)
        raise

    return run


# ---------- Derived aggregates ----------


def compute_run_counts(runs: list[AssessmentRun]) -> AssessmentRunCounts:
    """Aggregate child run statuses into counters."""
    return AssessmentRunCounts(
        total=len(runs),
        pending=sum(1 for r in runs if r.status == "pending"),
        processing=sum(1 for r in runs if r.status in {"processing", "in_progress"}),
        completed=sum(1 for r in runs if r.status == "completed"),
        failed=sum(1 for r in runs if r.status == "failed"),
    )


def derive_assessment_status(counts: AssessmentRunCounts) -> str:
    """Compute parent assessment status from child run counters."""
    if counts.total == 0:
        return "pending"
    if counts.completed == counts.total:
        return "completed"
    if counts.failed == counts.total:
        return "failed"
    if (
        counts.completed > 0
        and counts.failed > 0
        and counts.pending == 0
        and counts.processing == 0
    ):
        return "completed_with_errors"
    if counts.pending > 0 and counts.pending == counts.total:
        return "pending"
    return "processing"


def build_run_stats(runs: list[AssessmentRun]) -> list[AssessmentRunStat]:
    """Build per-run summary entries for embedding in parent responses."""
    return [
        AssessmentRunStat(
            run_id=r.id,
            config_id=str(r.config_id) if r.config_id else None,
            config_version=r.config_version,
            status=r.status,
            total_items=r.total_items,
            error_message=r.error_message,
            updated_at=r.updated_at,
        )
        for r in runs
    ]


def derive_aggregate_error(counts: AssessmentRunCounts) -> str | None:
    """Build an aggregate error summary string for parent assessments."""
    if counts.failed > 0:
        return f"{counts.failed} of {counts.total} run(s) failed"
    return None


def recompute_assessment_status(
    session: Session,
    assessment_id: int,
) -> Assessment:
    """Recompute the parent's `status` from its child runs.

    Counters and run_stats are derived on-read; only `status` is persisted so
    cron's `WHERE status IN (...)` filter remains index-friendly.
    """
    assessment = session.get(Assessment, assessment_id)
    if not assessment:
        raise ValueError(f"Assessment {assessment_id} not found")

    runs = get_assessment_runs_for_assessment(session, assessment_id)
    counts = compute_run_counts(runs)
    assessment.status = derive_assessment_status(counts)
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
