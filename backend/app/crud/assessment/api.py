"""Assessment API-client CRUD — writes for the BATCH pipeline on the `assessment` row.

Kept separate from the legacy RUN crud (core/cron/processing/batch), which is retired.
"""

import logging
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import cast as sa_cast
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, col, select

from app.core.util import now
from app.models.assessment import (
    Assessment,
    AssessmentExecution,
    AssessmentMethod,
    AssessmentStatus,
    AssessmentSubmission,
)

logger = logging.getLogger(__name__)


def create_assessment(
    *,
    session: Session,
    method: AssessmentMethod,
    config_id: UUID,
    config_version: int,
    total_items: int,
    organization_id: int,
    project_id: int,
    assessment_id: UUID | None = None,
    submission_id: UUID | None = None,
    submission_input: str | None = None,
    experiment_name: str | None = None,
    execution: AssessmentExecution | None = None,
) -> Assessment:
    """Insert the row; pass ``assessment_id`` when the object key already used it."""
    assessment = Assessment(
        id=assessment_id or uuid4(),
        method=method,
        config_id=config_id,
        config_version=config_version,
        total_items=total_items,
        execution=execution.model_dump(mode="json") if execution else None,
        submission_id=submission_id,
        submission_input=submission_input,
        experiment_name=experiment_name,
        status=AssessmentStatus.PENDING,
        organization_id=organization_id,
        project_id=project_id,
    )
    session.add(assessment)
    session.commit()
    session.refresh(assessment)
    logger.info(
        f"[create_assessment] Created | assessment_id: {assessment.id} | "
        f"method: {method} | config: {config_id} v{config_version} | "
        f"org: {organization_id} | project: {project_id}"
    )
    return assessment


def set_assessment_job(
    *, session: Session, assessment: Assessment, job_id: UUID
) -> Assessment:
    assessment.job_id = job_id
    assessment.updated_at = now()
    session.add(assessment)
    session.commit()
    session.refresh(assessment)
    logger.info(
        f"[set_assessment_job] Linked job | assessment_id: {assessment.id} | job_id: {job_id}"
    )
    return assessment


def save_execution_state(
    *, session: Session, assessment: Assessment, state: AssessmentExecution
) -> Assessment:
    """Persist the whole runtime bag. Reassigned + flagged: JSONB in-place edits are invisible to SQLAlchemy."""
    assessment.execution = state.model_dump(mode="json")
    flag_modified(assessment, "execution")
    assessment.updated_at = now()
    session.add(assessment)
    session.commit()
    session.refresh(assessment)
    logger.info(
        f"[save_execution_state] Saved | assessment_id: {assessment.id} | "
        f"stage: {state.stage} | stage_status: {state.stage_status}"
    )
    return assessment


def load_execution_state(assessment: Assessment) -> AssessmentExecution | None:
    return (
        AssessmentExecution.model_validate(assessment.execution)
        if assessment.execution
        else None
    )


def set_result_files(
    *, session: Session, assessment: Assessment, files: dict[str, dict[str, Any]]
) -> Assessment:
    """Shallow-merge ``files`` into ``assessment.result_files``, one record per file kind.

    Server-side ``||`` (right-hand wins per key): two drivers can touch this row within
    the same second, and a read-modify-write would drop the loser's kinds.
    """
    statement = (
        update(Assessment)
        .where(col(Assessment.id) == assessment.id)
        .values(
            result_files=col(Assessment.result_files).op("||")(sa_cast(files, JSONB)),
            updated_at=now(),
        )
    )
    try:
        session.exec(statement)
        session.commit()
    except Exception:
        session.rollback()
        raise
    session.refresh(assessment)
    logger.info(
        f"[set_result_files] Merged result files | assessment_id: {assessment.id} | "
        f"kinds: {sorted(files)}"
    )
    return assessment


def update_status(
    *, session: Session, assessment: Assessment, status: AssessmentStatus
) -> Assessment:
    assessment.status = status
    assessment.updated_at = now()
    session.add(assessment)
    session.commit()
    session.refresh(assessment)
    logger.info(
        f"[update_status] Updated | assessment_id: {assessment.id} | status: {status}"
    )
    return assessment


def list_assessments(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    config_id: UUID | None = None,
    config_version: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[tuple[Assessment, str | None]]:
    """BATCH assessments newest-first with their submission name (outer join: inline BATCH has none)."""
    statement = (
        select(Assessment, AssessmentSubmission.name)
        .join(
            AssessmentSubmission,
            col(AssessmentSubmission.id) == col(Assessment.submission_id),
            isouter=True,
        )
        .where(Assessment.method == AssessmentMethod.BATCH)
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
    )
    if config_id is not None:
        statement = statement.where(Assessment.config_id == config_id)
        if config_version is not None:
            statement = statement.where(Assessment.config_version == config_version)

    statement = (
        statement.order_by(
            col(Assessment.inserted_at).desc(), col(Assessment.id).desc()
        )
        .limit(limit)
        .offset(offset)
    )
    return list(session.exec(statement).all())
