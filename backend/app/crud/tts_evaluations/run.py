"""CRUD operations for TTS evaluation runs."""

import logging
from typing import Any

from sqlmodel import Session, func, select

from app.core.util import now
from app.models import EvaluationRun
from app.models.stt_evaluation import EvaluationType
from app.models.tts_evaluation import TTSEvaluationRunPublic

logger = logging.getLogger(__name__)


def create_tts_run(
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
    """Create a new TTS evaluation run.

    Args:
        session: Database session
        run_name: Name for the run
        dataset_id: ID of the dataset to evaluate
        dataset_name: Name of the dataset
        org_id: Organization ID
        project_id: Project ID
        models: List of TTS models to use
        language_id: Optional language ID override
        total_items: Total number of items to process

    Returns:
        EvaluationRun: Created run
    """
    logger.info(
        f"[create_tts_run] Creating TTS evaluation run | "
        f"run_name: {run_name}, dataset_id: {dataset_id}, "
        f"models: {models}"
    )

    run = EvaluationRun(
        run_name=run_name,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        type=EvaluationType.TTS.value,
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
        f"[create_tts_run] TTS evaluation run created | "
        f"run_id: {run.id}, run_name: {run_name}"
    )

    return run


def get_tts_run_by_id(
    *,
    session: Session,
    run_id: int,
    org_id: int,
    project_id: int,
) -> EvaluationRun | None:
    """Get a TTS evaluation run by ID.

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
        EvaluationRun.type == EvaluationType.TTS.value,
    )

    return session.exec(statement).one_or_none()


def list_tts_runs(
    *,
    session: Session,
    org_id: int,
    project_id: int,
    dataset_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[TTSEvaluationRunPublic], int]:
    """List TTS evaluation runs for a project.

    Args:
        session: Database session
        org_id: Organization ID
        project_id: Project ID
        dataset_id: Optional filter by dataset
        status: Optional filter by status
        limit: Maximum results to return
        offset: Number of results to skip

    Returns:
        tuple[list[TTSEvaluationRunPublic], int]: Runs and total count
    """
    where_clauses = [
        EvaluationRun.organization_id == org_id,
        EvaluationRun.project_id == project_id,
        EvaluationRun.type == EvaluationType.TTS.value,
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

    result = [TTSEvaluationRunPublic.from_model(run) for run in runs]

    return result, total


def update_tts_run(
    *,
    session: Session,
    run_id: int,
    status: str | None = None,
    score: dict[str, Any] | None = None,
    error_message: str | None = None,
    object_store_url: str | None = None,
    batch_job_id: int | None = None,
) -> EvaluationRun | None:
    """Update a TTS evaluation run.

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
        f"[update_tts_run] TTS run updated | run_id: {run_id}, status: {run.status}"
    )

    return run
