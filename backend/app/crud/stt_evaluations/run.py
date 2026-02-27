"""CRUD operations for STT evaluation runs."""

import logging
from typing import Any

from sqlmodel import Session, select, func

from app.core.util import now
from app.models import EvaluationRun
from app.models.stt_evaluation import (
    EvaluationType,
    STTEvaluationRunPublic,
)

logger = logging.getLogger(__name__)


def create_stt_run(
    *,
    session: Session,
    run_name: str,
    dataset_id: int,
    dataset_name: str,
    org_id: int,
    project_id: int,
    models: list[str],
    language_id: int | None = None,
    total_items: int = 0,
) -> EvaluationRun:
    """Create a new STT evaluation run.

    Args:
        session: Database session
        run_name: Name for the run
        dataset_id: ID of the dataset to evaluate
        dataset_name: Name of the dataset
        org_id: Organization ID
        project_id: Project ID
        models: List of STT models to use
        language_id: Optional language ID override (references global.languages)
        total_items: Total number of items to process

    Returns:
        EvaluationRun: Created run
    """
    logger.info(
        f"[create_stt_run] Creating STT evaluation run | "
        f"run_name: {run_name}, dataset_id: {dataset_id}, "
        f"models: {models}"
    )

    run = EvaluationRun(
        run_name=run_name,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        type=EvaluationType.STT.value,
        language_id=language_id,
        providers=models,
        status="pending",
        total_items=total_items,
        organization_id=org_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )

    session.add(run)
    session.commit()
    session.refresh(run)

    logger.info(
        f"[create_stt_run] STT evaluation run created | "
        f"run_id: {run.id}, run_name: {run_name}"
    )

    return run


def get_stt_run_by_id(
    *,
    session: Session,
    run_id: int,
    org_id: int,
    project_id: int,
) -> EvaluationRun | None:
    """Get an STT evaluation run by ID.

    Args:
        session: Database session
        run_id: Run ID
        org_id: Organization ID
        project_id: Project ID

    Returns:
        EvaluationRun | None: Run if found
    """
    statement = select(EvaluationRun).where(
        EvaluationRun.id == run_id,
        EvaluationRun.organization_id == org_id,
        EvaluationRun.project_id == project_id,
        EvaluationRun.type == EvaluationType.STT.value,
    )

    return session.exec(statement).one_or_none()


def list_stt_runs(
    *,
    session: Session,
    org_id: int,
    project_id: int,
    dataset_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[STTEvaluationRunPublic], int]:
    """List STT evaluation runs for a project.

    Args:
        session: Database session
        org_id: Organization ID
        project_id: Project ID
        dataset_id: Optional filter by dataset
        status: Optional filter by status
        limit: Maximum results to return
        offset: Number of results to skip

    Returns:
        tuple[list[STTEvaluationRunPublic], int]: Runs and total count
    """
    where_clauses = [
        EvaluationRun.organization_id == org_id,
        EvaluationRun.project_id == project_id,
        EvaluationRun.type == EvaluationType.STT.value,
    ]

    if dataset_id is not None:
        where_clauses.append(EvaluationRun.dataset_id == dataset_id)

    if status is not None:
        where_clauses.append(EvaluationRun.status == status)

    count_stmt = select(func.count(EvaluationRun.id)).where(*where_clauses)
    total = session.exec(count_stmt).one()

    statement = (
        select(EvaluationRun)
        .where(*where_clauses)
        .order_by(EvaluationRun.inserted_at.desc())
        .offset(offset)
        .limit(limit)
    )

    runs = session.exec(statement).all()

    result = [
        STTEvaluationRunPublic(
            id=run.id,
            run_name=run.run_name,
            dataset_name=run.dataset_name,
            type=run.type,
            language_id=run.language_id,
            models=run.providers,
            dataset_id=run.dataset_id,
            status=run.status,
            total_items=run.total_items,
            score=run.score,
            error_message=run.error_message,
            organization_id=run.organization_id,
            project_id=run.project_id,
            inserted_at=run.inserted_at,
            updated_at=run.updated_at,
        )
        for run in runs
    ]

    return result, total


def update_stt_run(
    *,
    session: Session,
    run_id: int,
    status: str | None = None,
    score: dict[str, Any] | None = None,
    error_message: str | None = None,
    object_store_url: str | None = None,
    batch_job_id: int | None = None,
) -> EvaluationRun | None:
    """Update an STT evaluation run.

    Args:
        session: Database session
        run_id: Run ID
        status: New status
        score: Score data
        error_message: Error message
        object_store_url: URL to stored results
        batch_job_id: ID of the associated batch job

    Returns:
        EvaluationRun | None: Updated run
    """
    statement = select(EvaluationRun).where(EvaluationRun.id == run_id)
    run = session.exec(statement).one_or_none()

    if not run:
        return None

    updates = {
        "status": status,
        "score": score,
        "error_message": error_message,
        "object_store_url": object_store_url,
        "batch_job_id": batch_job_id,
    }

    for field, value in updates.items():
        if value is not None:
            setattr(run, field, value)

    run.updated_at = now()

    session.add(run)
    session.commit()
    session.refresh(run)

    logger.info(
        f"[update_stt_run] STT run updated | run_id: {run_id}, status: {run.status}"
    )

    return run
