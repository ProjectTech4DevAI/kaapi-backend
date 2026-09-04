"""Assessment API-client CRUD — method-based Assessment / AssessmentRun writes.

Kept separate from the legacy RUN-pipeline crud (core/cron/processing/batch):
writes only the new method-based columns and leaves the RUN-only `execution`
and `dataset_id` fields NULL.
"""

import logging
from typing import Any, TypeVar, cast
from uuid import UUID

from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select

from app.core.util import now
from app.models.assessment import (
    Assessment,
    AssessmentMethod,
    AssessmentRun,
    AssessmentStatus,
    BatchRunState,
)

logger = logging.getLogger(__name__)

# update_status works on either status-bearing row; the TypeVar preserves which.
StatusModel = TypeVar("StatusModel", Assessment, AssessmentRun)


def create_assessment(
    *,
    session: Session,
    method: AssessmentMethod,
    input: dict[str, Any],
    organization_id: int,
    project_id: int,
) -> Assessment:
    assessment = Assessment(
        method=method,
        input=input,
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


def list_executions(*, session: Session, assessment_id: UUID) -> list[AssessmentRun]:
    statement = (
        select(AssessmentRun)
        .where(AssessmentRun.assessment_id == assessment_id)
        .order_by(AssessmentRun.id.asc())
    )
    return list(session.exec(statement).all())
