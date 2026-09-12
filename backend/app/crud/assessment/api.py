"""Assessment API-client CRUD — method-based Assessment / AssessmentRun writes.

Kept separate from the UI-only crud (core/cron/processing/batch):
writes only the new method-based columns and leaves the RUN-only `execution`
and `submission_id` fields NULL.
"""

import logging
from typing import Any, TypeVar, cast
from uuid import UUID

from sqlalchemy import cast as sa_cast
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, col, select

from app.core.util import now
from app.models.assessment import (
    Assessment,
    AssessmentMethod,
    AssessmentRun,
    AssessmentStatus,
    AssessmentSubmission,
    BatchRunState,
)

logger = logging.getLogger(__name__)

# update_status works on either status-bearing row; the TypeVar preserves which.
StatusModel = TypeVar("StatusModel", Assessment, AssessmentRun)


def create_assessment(
    *,
    session: Session,
    method: AssessmentMethod,
    input: dict[str, Any] | None,
    organization_id: int,
    project_id: int,
    submission_id: UUID | None = None,
    experiment_name: str | None = None,
) -> Assessment:
    assessment = Assessment(
        method=method,
        input=input,
        submission_id=submission_id,
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
        f"method: {method} | org: {organization_id} | project: {project_id}"
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


def set_submission_input(
    *, session: Session, assessment: Assessment, url: str
) -> Assessment:
    """Point the assessment at its stored submission rows."""
    assessment.submission_input = url
    assessment.updated_at = now()
    session.add(assessment)
    session.commit()
    session.refresh(assessment)
    logger.info(
        f"[set_submission_input] Linked submission | assessment_id: {assessment.id} | url: {url}"
    )
    return assessment


def create_execution(
    *,
    session: Session,
    assessment_id: UUID,
    config_id: UUID,
    config_version: int,
    total_items: int,
) -> AssessmentRun:
    execution = AssessmentRun(
        assessment_id=assessment_id,
        config_id=config_id,
        config_version=config_version,
        status=AssessmentStatus.PENDING,
        total_items=total_items,
    )
    session.add(execution)
    session.commit()
    session.refresh(execution)
    logger.info(
        f"[create_execution] Created | execution_id: {execution.id} | "
        f"assessment_id: {assessment_id} | config_id: {config_id} v{config_version}"
    )
    return execution


def set_execution_batch_job(
    *, session: Session, execution: AssessmentRun, batch_job_id: int
) -> AssessmentRun:
    execution.batch_job_id = batch_job_id
    execution.updated_at = now()
    session.add(execution)
    session.commit()
    session.refresh(execution)
    logger.info(
        f"[set_execution_batch_job] Linked batch job | execution_id: {execution.id} | "
        f"batch_job_id: {batch_job_id}"
    )
    return execution


def save_execution_state(
    *, session: Session, execution: AssessmentRun, state: BatchRunState
) -> AssessmentRun:
    """Persist the whole staged-batch runtime bag onto ``execution.execution``.

    JSONB in-place mutation is invisible to SQLAlchemy, so we reassign the column
    and flag it modified rather than mutating the existing dict.
    """
    # mypy treats a TypedDict as incompatible with the column's plain dict[str, Any];
    # the cast is erased at runtime (a TypedDict already is a dict).
    execution.execution = cast(dict[str, Any], state)
    flag_modified(execution, "execution")
    execution.updated_at = now()
    session.add(execution)
    session.commit()
    session.refresh(execution)
    logger.info(
        f"[save_execution_state] Saved | execution_id: {execution.id} | "
        f"stage: {state.get('stage')} | stage_status: {state.get('stage_status')}"
    )
    return execution


def set_result_files(
    *, session: Session, assessment: Assessment, files: dict[str, dict[str, Any]]
) -> Assessment:
    """Shallow-merge ``files`` into ``assessment.result_files``, one record per file kind.

    The merge is server-side (``||``, right-hand side wins per key) because two drivers
    can touch this row within the same second; a read-modify-write would drop the loser's
    kinds instead of keeping both.
    """
    statement = (
        update(Assessment)
        .where(col(Assessment.id) == assessment.id)
        .values(
            result_files=col(Assessment.result_files).op("||")(sa_cast(files, JSONB)),
            updated_at=now(),
        )
    )
    session.exec(statement)
    session.commit()
    session.refresh(assessment)
    logger.info(
        f"[set_result_files] Merged result files | assessment_id: {assessment.id} | "
        f"kinds: {sorted(files)}"
    )
    return assessment


def update_status(
    *, session: Session, obj: StatusModel, status: AssessmentStatus
) -> StatusModel:
    """Set status on an Assessment or AssessmentRun; both carry `status`/`updated_at`."""
    obj.status = status
    obj.updated_at = now()
    session.add(obj)
    session.commit()
    session.refresh(obj)
    logger.info(
        f"[update_status] Updated | {type(obj).__name__}: {obj.id} | status: {status}"
    )
    return obj


def list_assessments_with_execution(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    method: AssessmentMethod | None = None,
    config_id: UUID | None = None,
    config_version: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[tuple[Assessment, AssessmentRun | None, str | None]]:
    """Assessments newest-first, each with its execution and submission name.

    Outer joins throughout: an inline BATCH has no submission, and an assessment
    whose execution insert failed should still list. ``config_id``/``config_version``
    filter on the execution, which is where the config pin lives.
    """
    statement = (
        select(Assessment, AssessmentRun, AssessmentSubmission.name)
        .join(
            AssessmentRun,
            col(AssessmentRun.assessment_id) == col(Assessment.id),
            isouter=True,
        )
        .join(
            AssessmentSubmission,
            col(AssessmentSubmission.id) == col(Assessment.submission_id),
            isouter=True,
        )
        .where(Assessment.organization_id == organization_id)
        .where(Assessment.project_id == project_id)
    )
    if method is not None:
        statement = statement.where(Assessment.method == method)
    if config_id is not None:
        statement = statement.where(AssessmentRun.config_id == config_id)
        if config_version is not None:
            statement = statement.where(AssessmentRun.config_version == config_version)

    statement = (
        statement.order_by(col(Assessment.inserted_at).desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.exec(statement).all())


def list_executions(*, session: Session, assessment_id: UUID) -> list[AssessmentRun]:
    statement = (
        select(AssessmentRun)
        .where(AssessmentRun.assessment_id == assessment_id)
        .order_by(AssessmentRun.id.asc())
    )
    return list(session.exec(statement).all())
