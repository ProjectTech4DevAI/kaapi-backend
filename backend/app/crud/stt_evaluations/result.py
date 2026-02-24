"""CRUD operations for STT evaluation results."""

import logging

from sqlmodel import Session, select, func

from app.core.exception_handlers import HTTPException
from app.core.util import now
from app.models.file import File
from app.models.stt_evaluation import (
    STTResult,
    STTSample,
    STTSamplePublic,
    STTResultWithSample,
)

logger = logging.getLogger(__name__)


def get_stt_result_by_id(
    *,
    session: Session,
    result_id: int,
    org_id: int,
    project_id: int,
) -> STTResult | None:
    """Get an STT result by ID.

    Args:
        session: Database session
        result_id: Result ID
        org_id: Organization ID
        project_id: Project ID

    Returns:
        STTResult | None: Result if found
    """
    statement = select(STTResult).where(
        STTResult.id == result_id,
        STTResult.organization_id == org_id,
        STTResult.project_id == project_id,
    )

    return session.exec(statement).one_or_none()


def get_results_by_run_id(
    *,
    session: Session,
    run_id: int,
    org_id: int,
    project_id: int,
    provider: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[STTResultWithSample], int]:
    """Get results for an evaluation run with sample data.

    Args:
        session: Database session
        run_id: Run ID
        org_id: Organization ID
        project_id: Project ID
        provider: Optional filter by provider
        status: Optional filter by status
        limit: Maximum results to return
        offset: Number of results to skip

    Returns:
        tuple[list[STTResultWithSample], int]: Results with samples and total count
    """
    where_clauses = [
        STTResult.evaluation_run_id == run_id,
        STTResult.organization_id == org_id,
        STTResult.project_id == project_id,
    ]

    if provider is not None:
        where_clauses.append(STTResult.provider == provider)

    if status is not None:
        where_clauses.append(STTResult.status == status)

    count_stmt = select(func.count(STTResult.id)).where(*where_clauses)
    total = session.exec(count_stmt).one()

    statement = (
        select(STTResult, STTSample, File)
        .join(STTSample, STTResult.stt_sample_id == STTSample.id)
        .join(File, STTSample.file_id == File.id)
        .where(*where_clauses)
        .order_by(STTResult.id)
        .offset(offset)
        .limit(limit)
    )

    rows = session.exec(statement).all()

    # Convert to response models
    results = []
    for result, sample, file in rows:
        sample_public = STTSamplePublic(
            id=sample.id,
            file_id=sample.file_id,
            object_store_url=file.object_store_url,
            language_id=sample.language_id,
            ground_truth=sample.ground_truth,
            sample_metadata=sample.sample_metadata,
            dataset_id=sample.dataset_id,
            organization_id=sample.organization_id,
            project_id=sample.project_id,
            inserted_at=sample.inserted_at,
            updated_at=sample.updated_at,
        )

        result_with_sample = STTResultWithSample(
            id=result.id,
            transcription=result.transcription,
            provider=result.provider,
            status=result.status,
            score=result.score,
            is_correct=result.is_correct,
            comment=result.comment,
            error_message=result.error_message,
            stt_sample_id=result.stt_sample_id,
            evaluation_run_id=result.evaluation_run_id,
            organization_id=result.organization_id,
            project_id=result.project_id,
            inserted_at=result.inserted_at,
            updated_at=result.updated_at,
            sample=sample_public,
        )
        results.append(result_with_sample)

    return results, total


def update_human_feedback(
    *,
    session: Session,
    result_id: int,
    org_id: int,
    project_id: int,
    is_correct: bool | None = None,
    comment: str | None = None,
) -> STTResult | None:
    """Update human feedback on an STT result.

    Args:
        session: Database session
        result_id: Result ID
        org_id: Organization ID
        project_id: Project ID
        is_correct: Human verification of correctness
        comment: Feedback comment

    Returns:
        STTResult | None: Updated result

    Raises:
        HTTPException: If result not found
    """
    result = get_stt_result_by_id(
        session=session,
        result_id=result_id,
        org_id=org_id,
        project_id=project_id,
    )

    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    if is_correct is not None:
        result.is_correct = is_correct

    if comment is not None:
        result.comment = comment

    result.updated_at = now()

    session.add(result)
    session.commit()
    session.refresh(result)

    logger.info(
        f"[update_human_feedback] Human feedback updated | "
        f"result_id: {result_id}, is_correct: {is_correct}"
    )

    return result


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
        select(STTResult.status, func.count(STTResult.id))
        .where(STTResult.evaluation_run_id == run_id)
        .group_by(STTResult.status)
    )

    rows = session.exec(statement).all()

    return {status: count for status, count in rows}
