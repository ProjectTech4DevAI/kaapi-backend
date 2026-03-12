"""CRUD operations for TTS evaluation results."""

import logging
from typing import Any

from sqlmodel import Session, func, select

from app.core.cloud.storage import CloudStorage
from app.core.exception_handlers import HTTPException
from app.core.util import now
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult, TTSResultPublic

logger = logging.getLogger(__name__)


def create_tts_results(
    *,
    session: Session,
    sample_texts: list[str],
    evaluation_run_id: int,
    org_id: int,
    project_id: int,
    models: list[str],
) -> list[TTSResult]:
    """Create TTS result records for all sample texts and models.

    Creates one result per sample text per model.

    Args:
        session: Database session
        sample_texts: List of text strings to synthesize
        evaluation_run_id: Run ID
        org_id: Organization ID
        project_id: Project ID
        models: List of TTS models

    Returns:
        list[TTSResult]: Created results
    """
    logger.info(
        f"[create_tts_results] Creating TTS results | "
        f"run_id: {evaluation_run_id}, sample_count: {len(sample_texts)}, "
        f"model_count: {len(models)}"
    )

    timestamp = now()
    results = [
        TTSResult(
            sample_text=text,
            evaluation_run_id=evaluation_run_id,
            organization_id=org_id,
            project_id=project_id,
            provider=model,
            status=JobStatus.PENDING.value,
            inserted_at=timestamp,
            updated_at=timestamp,
        )
        for text in sample_texts
        for model in models
    ]

    session.add_all(results)
    session.flush()
    session.commit()

    logger.info(
        f"[create_tts_results] TTS results created | "
        f"run_id: {evaluation_run_id}, result_count: {len(results)}"
    )

    return results


def get_tts_result_by_id(
    *,
    session: Session,
    result_id: int,
    org_id: int,
    project_id: int,
) -> TTSResult | None:
    """Get a TTS result by ID.

    Args:
        session: Database session
        result_id: Result ID
        org_id: Organization ID
        project_id: Project ID

    Returns:
        TTSResult | None: Result if found
    """
    statement = select(TTSResult).where(
        TTSResult.id == result_id,
        TTSResult.organization_id == org_id,
        TTSResult.project_id == project_id,
    )

    return session.exec(statement).one_or_none()


def get_results_by_run_id(
    *,
    session: Session,
    run_id: int,
    org_id: int,
    project_id: int,
    storage: CloudStorage | None = None,
) -> tuple[list[TTSResultPublic], int]:
    """Get all results for an evaluation run.

    Args:
        session: Database session
        run_id: Run ID
        org_id: Organization ID
        project_id: Project ID
        storage: Optional cloud storage instance for generating signed URLs

    Returns:
        tuple[list[TTSResultPublic], int]: Results and total count
    """
    where_clauses = [
        TTSResult.evaluation_run_id == run_id,
        TTSResult.organization_id == org_id,
        TTSResult.project_id == project_id,
    ]

    statement = select(TTSResult).where(*where_clauses).order_by(TTSResult.id)

    rows = session.exec(statement).all()
    total = len(rows)

    results = []
    for result in rows:
        signed_url = (
            storage.get_signed_url(result.object_store_url) if storage else None
        )
        results.append(TTSResultPublic.from_model(result, signed_url=signed_url))

    return results, total


def update_tts_result(
    *,
    session: Session,
    result_id: int,
    org_id: int | None = None,
    project_id: int | None = None,
    object_store_url: str | None = None,
    metadata: dict[str, Any] | None = None,
    status: str | None = None,
    error_message: str | None = None,
) -> TTSResult | None:
    """Update a TTS result.

    Args:
        session: Database session
        result_id: Result ID
        org_id: Organization ID (optional, for scoping)
        project_id: Project ID (optional, for scoping)
        object_store_url: S3 URL of generated WAV
        metadata: Audio metadata (duration_seconds, size_bytes)
        status: New status
        error_message: Error message if failed

    Returns:
        TTSResult | None: Updated result
    """
    where_clauses = [TTSResult.id == result_id]
    if org_id is not None:
        where_clauses.append(TTSResult.organization_id == org_id)
    if project_id is not None:
        where_clauses.append(TTSResult.project_id == project_id)

    statement = select(TTSResult).where(*where_clauses)
    result = session.exec(statement).one_or_none()

    if not result:
        return None

    if object_store_url is not None:
        result.object_store_url = object_store_url
    if metadata is not None:
        result.metadata_ = metadata
    if status is not None:
        result.status = status
    if error_message is not None:
        result.error_message = error_message

    result.updated_at = now()

    session.add(result)
    session.flush()

    return result


def update_tts_human_feedback(
    *,
    session: Session,
    result_id: int,
    org_id: int,
    project_id: int,
    **kwargs: Any,
) -> TTSResult | None:
    """Update human feedback on a TTS result.

    Only fields passed in kwargs are updated. Passing is_correct=None clears
    the value; omitting it leaves it unchanged.

    Args:
        session: Database session
        result_id: Result ID
        org_id: Organization ID
        project_id: Project ID
        **kwargs: Fields to update (is_correct, comment)

    Returns:
        TTSResult | None: Updated result

    Raises:
        HTTPException: If result not found
    """
    result = get_tts_result_by_id(
        session=session,
        result_id=result_id,
        org_id=org_id,
        project_id=project_id,
    )

    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    if "is_correct" in kwargs:
        result.is_correct = kwargs["is_correct"]

    if "comment" in kwargs:
        result.comment = kwargs["comment"]

    result.updated_at = now()

    session.add(result)
    session.commit()
    session.refresh(result)

    logger.info(
        f"[update_tts_human_feedback] Human feedback updated | "
        f"result_id: {result_id}, is_correct: {kwargs.get('is_correct')}"
    )

    return result


def get_pending_results_for_run(
    *,
    session: Session,
    run_id: int,
    provider: str | None = None,
) -> list[TTSResult]:
    """Get all pending results for a run.

    Args:
        session: Database session
        run_id: Run ID
        provider: Optional filter by provider

    Returns:
        list[TTSResult]: Pending results
    """
    where_clauses = [
        TTSResult.evaluation_run_id == run_id,
        TTSResult.status == JobStatus.PENDING.value,
    ]

    if provider is not None:
        where_clauses.append(TTSResult.provider == provider)

    statement = select(TTSResult).where(*where_clauses)

    return list(session.exec(statement).all())


def count_results_by_status(
    *,
    session: Session,
    run_id: int,
) -> dict[str, int]:
    """Count results by status for a run.

    Args:
        session: Database session
        run_id: Run ID

    Returns:
        dict[str, int]: Counts by status
    """
    statement = (
        select(TTSResult.status, func.count(TTSResult.id))
        .where(TTSResult.evaluation_run_id == run_id)
        .group_by(TTSResult.status)
    )

    rows = session.exec(statement).all()

    return {status: count for status, count in rows}
