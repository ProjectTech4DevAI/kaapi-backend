"""CRUD for the thin `evaluation_iteration_run` tracking row.

The round-by-round trajectory lives in the LangGraph checkpoint, not here — this
module only creates/looks up/updates the one thin row per loop.
"""

import logging
from uuid import UUID

from sqlmodel import Session, select

from app.core.util import now
from app.models.evaluation_iteration import (
    EvaluationIterationRun,
    EvaluationIterationRunUpdate,
    EvaluationIterationStatusEnum,
)

logger = logging.getLogger(__name__)


def create_evaluation_iteration_run(
    *,
    session: Session,
    dataset_id: int,
    experiment_name: str,
    config_id: UUID,
    initial_config_version: int,
    callback_url: str,
    organization_id: int,
    project_id: int,
) -> EvaluationIterationRun:
    """Create the thin tracking row, status=PROCESSING."""
    iteration_run = EvaluationIterationRun(
        dataset_id=dataset_id,
        experiment_name=experiment_name,
        config_id=config_id,
        initial_config_version=initial_config_version,
        callback_url=callback_url,
        status=EvaluationIterationStatusEnum.PROCESSING,
        organization_id=organization_id,
        project_id=project_id,
    )
    session.add(iteration_run)
    session.commit()
    session.refresh(iteration_run)
    logger.info(
        f"[create_evaluation_iteration_run] Created | "
        f"iteration_run_id={iteration_run.id} | dataset_id={dataset_id} | "
        f"org_id={organization_id} | project_id={project_id}"
    )
    return iteration_run


def get_evaluation_iteration_run_by_id(
    *,
    session: Session,
    iteration_run_id: int,
    organization_id: int,
    project_id: int,
) -> EvaluationIterationRun | None:
    """Get one iteration run scoped to the caller's org/project."""
    statement = (
        select(EvaluationIterationRun)
        .where(EvaluationIterationRun.id == iteration_run_id)
        .where(EvaluationIterationRun.organization_id == organization_id)
        .where(EvaluationIterationRun.project_id == project_id)
    )
    iteration_run = session.exec(statement).first()
    if iteration_run is None:
        logger.warning(
            f"[get_evaluation_iteration_run_by_id] Not found | "
            f"iteration_run_id={iteration_run_id} | org_id={organization_id} | "
            f"project_id={project_id}"
        )
    return iteration_run


def list_processing_evaluation_iteration_runs(
    *, session: Session
) -> list[EvaluationIterationRun]:
    """Every loop still in flight — what the cron tick dispatches a resume for."""
    statement = select(EvaluationIterationRun).where(
        EvaluationIterationRun.status == EvaluationIterationStatusEnum.PROCESSING
    )
    runs = list(session.exec(statement).all())
    logger.info(
        f"[list_processing_evaluation_iteration_runs] Found {len(runs)} processing loops"
    )
    return runs


def update_evaluation_iteration_run(
    *,
    session: Session,
    iteration_run: EvaluationIterationRun,
    update: EvaluationIterationRunUpdate,
) -> EvaluationIterationRun:
    """Partial update; only fields explicitly set on `update` are applied."""
    update_fields = update.model_dump(exclude_unset=True)
    for field_name, new_value in update_fields.items():
        setattr(iteration_run, field_name, new_value)

    iteration_run.updated_at = now()
    session.add(iteration_run)
    session.commit()
    session.refresh(iteration_run)
    logger.info(
        f"[update_evaluation_iteration_run] Updated | "
        f"iteration_run_id={iteration_run.id} | fields={list(update_fields.keys())}"
    )
    return iteration_run
