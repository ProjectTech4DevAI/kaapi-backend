"""Assessment CRUD — operations for Assessment and AssessmentRun tables."""

import logging
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select

from app.core.util import now
from app.models.assessment import (
    Assessment,
    AssessmentMethod,
    AssessmentRun,
    AssessmentRunCounts,
    AssessmentRunStat,
    AssessmentStatus,
)

logger = logging.getLogger(__name__)


def _read_exec(run: AssessmentRun) -> dict[str, Any]:
    """Read the RUN runtime bag (RunExecution shape) that replaced the dropped columns."""
    return run.execution or {}


def _write_exec(run: AssessmentRun, **values: Any) -> None:
    """Merge keys into the run's execution bag; the caller commits."""
    run.execution = {**(run.execution or {}), **values}
    flag_modified(run, "execution")


def create_assessment(
    session: Session,
    experiment_name: str,
    dataset_id: int,
    organization_id: int,
    project_id: int,
    input_binding: dict[str, Any] | None = None,
) -> Assessment:
    """Create a parent assessment row for the legacy RUN pipeline.

    The RUN binding (InputBinding: prompt/text_columns/attachments) now lives on
    the parent `assessment.input`; child runs no longer carry their own input.
    """
    assessment = Assessment(
        experiment_name=experiment_name,
        method=AssessmentMethod.RUN,
        dataset_id=dataset_id,
        input=input_binding,
        status=AssessmentStatus.PENDING,
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
    assessment_id: UUID,
    organization_id: int,
    project_id: int,
) -> Assessment:
    """Get a specific parent assessment row."""
    statement = (
        select(Assessment)
        .where(Assessment.id == assessment_id)
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
    )
    assessment = session.exec(statement).first()
    if not assessment:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment {assessment_id} not found or not accessible",
        )
    return assessment


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
    assessment_id: UUID,
    config_id: UUID,
    config_version: int,
) -> AssessmentRun:
    """Create an assessment run record under a parent assessment.

    The run's input binding lives on the parent `assessment.input`; the run row
    carries only config + runtime state.
    """
    run = AssessmentRun(
        assessment_id=assessment_id,
        config_id=config_id,
        config_version=config_version,
        status=AssessmentStatus.PENDING,
        total_items=0,
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


def update_run_post_processing_config(
    session: Session,
    run: AssessmentRun,
    config: dict[str, Any] | None,
) -> AssessmentRun:
    """Persist the run's post_processing_config column."""
    run.post_processing_config = config
    run.updated_at = now()
    session.add(run)
    try:
        session.commit()
        session.refresh(run)
    except Exception as e:
        session.rollback()
        logger.error(
            f"[update_run_post_processing_config] Failed for run id={run.id}: {e}",
            exc_info=True,
        )
        raise

    logger.info(f"[update_run_post_processing_config] Updated run id={run.id}")
    return run


def get_assessment_run_by_id(
    session: Session,
    run_id: int,
    organization_id: int,
    project_id: int,
) -> AssessmentRun:
    """Get a specific assessment run by ID, scoped via parent organization/project."""
    statement = (
        select(AssessmentRun)
        .join(Assessment, Assessment.id == AssessmentRun.assessment_id)
        .where(AssessmentRun.id == run_id)
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
    )
    run = session.exec(statement).first()
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Assessment run {run_id} not found or not accessible",
        )
    return run


def get_assessment_runs_for_assessment(
    session: Session,
    assessment_id: UUID,
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
    assessment_id: UUID | None = None,
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
    status: AssessmentStatus,
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
        _write_exec(run, object_store_url=object_store_url)

    session.add(run)
    try:
        session.commit()
        session.refresh(run)
    except Exception as e:
        session.rollback()
        logger.error(f"[update_assessment_run_status] Failed: {e}", exc_info=True)
        raise

    return run


def update_assessment_run_prefilter_stats(
    session: Session,
    run: AssessmentRun,
    prefilter_object_store_url: str | None = None,
    prefilter_total_rows: int | None = None,
    prefilter_total_passed: int | None = None,
    prefilter_total_rejected: int | None = None,
) -> AssessmentRun:
    """Persist prefilter result stats (rows/passed/rejected + S3 URL) in the exec bag."""
    run.updated_at = now()

    stats: dict[str, Any] = {}
    if prefilter_object_store_url is not None:
        stats["prefilter_object_store_url"] = prefilter_object_store_url
    if prefilter_total_rows is not None:
        stats["prefilter_total_rows"] = prefilter_total_rows
    if prefilter_total_passed is not None:
        stats["prefilter_total_passed"] = prefilter_total_passed
    if prefilter_total_rejected is not None:
        stats["prefilter_total_rejected"] = prefilter_total_rejected
    if stats:
        _write_exec(run, **stats)

    session.add(run)
    try:
        session.commit()
        session.refresh(run)
    except Exception as e:
        session.rollback()
        logger.error(
            f"[update_assessment_run_prefilter_stats] Failed: {e}", exc_info=True
        )
        raise

    return run


# Stage-level progress (prefilter/l2) now lives in the exec bag's stage_status;
# run.status is one of the AssessmentStatus terminal/lifecycle values.
_COMPLETED_RUN_STATUSES = {
    AssessmentStatus.COMPLETED,
    AssessmentStatus.COMPLETED_WITH_ERRORS,
}


def compute_run_counts(runs: list[AssessmentRun]) -> AssessmentRunCounts:
    """Aggregate child run statuses into counters."""
    return AssessmentRunCounts(
        total=len(runs),
        pending=sum(1 for run in runs if run.status == AssessmentStatus.PENDING),
        processing=sum(1 for run in runs if run.status == AssessmentStatus.PROCESSING),
        completed=sum(1 for run in runs if run.status in _COMPLETED_RUN_STATUSES),
        failed=sum(1 for run in runs if run.status == AssessmentStatus.FAILED),
    )


def derive_assessment_status(counts: AssessmentRunCounts) -> AssessmentStatus:
    """Compute parent assessment status from child run counters."""
    if counts.total == 0:
        return AssessmentStatus.PENDING
    if counts.completed == counts.total:
        return AssessmentStatus.COMPLETED
    if counts.failed == counts.total:
        return AssessmentStatus.FAILED
    if (
        counts.completed > 0
        and counts.failed > 0
        and counts.pending == 0
        and counts.processing == 0
    ):
        return AssessmentStatus.COMPLETED_WITH_ERRORS
    if counts.pending > 0 and counts.pending == counts.total:
        return AssessmentStatus.PENDING
    return AssessmentStatus.PROCESSING


def build_run_stats(runs: list[AssessmentRun]) -> list[AssessmentRunStat]:
    """Build per-run summary entries for embedding in parent responses."""
    return [
        AssessmentRunStat(
            run_id=run.id,
            config_id=run.config_id,
            config_version=run.config_version,
            status=run.status,
            total_items=run.total_items,
            error_message=run.error_message,
            updated_at=run.updated_at,
        )
        for run in runs
    ]


def derive_aggregate_error(counts: AssessmentRunCounts) -> str | None:
    """Build an aggregate error summary string for parent assessments."""
    if counts.failed > 0:
        return f"{counts.failed} of {counts.total} run(s) failed"
    return None


def recompute_assessment_status(
    session: Session,
    assessment_id: UUID,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> Assessment:
    """Recompute the parent's `status` from its child runs.

    Counters and run_stats are derived on-read; only `status` is persisted so
    cron's `WHERE status IN (...)` filter remains index-friendly.
    """
    if organization_id is None and project_id is None:
        assessment = session.get(Assessment, assessment_id)
    else:
        statement = select(Assessment).where(Assessment.id == assessment_id)
        if organization_id is not None:
            statement = statement.where(Assessment.organization_id == organization_id)
        if project_id is not None:
            statement = statement.where(Assessment.project_id == project_id)
        assessment = session.exec(statement).first()
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
