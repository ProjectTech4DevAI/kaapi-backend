"""CRUD operations for TTS evaluation results."""

import logging
from typing import Any

from sqlalchemy import BigInteger, Float, update
from sqlmodel import Session, col, func, select

from app.core.cloud.storage import CloudStorage
from app.core.exception_handlers import HTTPException
from app.core.util import now
from app.models.job import JobStatus
from app.models.tts_evaluation import TTSResult, TTSResultPublic, TTSUsageUnitEnum

logger = logging.getLogger(__name__)

# Numeric usage fields summed per unit; credits are fractional, the rest are counts.
USAGE_SUM_FIELDS_BY_UNIT: dict[str, tuple[str, ...]] = {
    TTSUsageUnitEnum.TOKENS: ("input_tokens", "output_tokens", "total_tokens"),
    TTSUsageUnitEnum.CHARACTERS: ("characters",),
    TTSUsageUnitEnum.CREDITS: ("characters", "credits"),
}
FRACTIONAL_USAGE_FIELDS = {"credits"}


def create_tts_results(
    *,
    session: Session,
    sample_texts: list[str],
    evaluation_run_id: int,
    org_id: int,
    project_id: int,
    models: list[str],
    voices: dict[str, str] | None = None,
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
        voices: Voice per model; a model missing here gets a NULL voice, which
            readers treat as the model's default

    Returns:
        list[TTSResult]: Created results
    """
    logger.info(
        f"[create_tts_results] Creating TTS results | "
        f"run_id: {evaluation_run_id}, sample_count: {len(sample_texts)}, "
        f"model_count: {len(models)}"
    )

    model_voices = voices or {}
    timestamp = now()
    results: list[TTSResult] = []
    for text in sample_texts:
        for model in models:
            results.append(
                TTSResult(
                    sample_text=text,
                    evaluation_run_id=evaluation_run_id,
                    organization_id=org_id,
                    project_id=project_id,
                    provider=model,
                    voice=model_voices.get(model),
                    status=JobStatus.PENDING.value,
                    inserted_at=timestamp,
                    updated_at=timestamp,
                )
            )

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
            storage.get_signed_url(result.object_store_url)
            if storage and result.object_store_url
            else None
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
    usage: dict[str, Any] | None = None,
) -> TTSResult | None:
    """Update a TTS result that is still PENDING.

    The cron can mark a stale row FAILED and finalize the run while a sync worker
    is still synthesizing it; the PENDING guard drops that worker's late write so
    the row stays consistent with the finalized run.

    Args:
        session: Database session
        result_id: Result ID
        org_id: Organization ID (optional, for scoping)
        project_id: Project ID (optional, for scoping)
        object_store_url: S3 URL of generated WAV
        metadata: Audio metadata (duration_seconds, size_bytes)
        status: New status
        error_message: Error message if failed
        usage: Provider-native usage ({unit, ...})

    Returns:
        TTSResult | None: Updated result, or None if missing or no longer PENDING
    """
    where_clauses = [
        TTSResult.id == result_id,
        TTSResult.status == JobStatus.PENDING.value,
    ]
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
    if usage is not None:
        result.usage = usage

    result.updated_at = now()

    session.add(result)
    session.flush()

    return result


def fail_pending_tts_results(
    *,
    session: Session,
    run_id: int,
    provider: str,
    error_message: str,
) -> int:
    """Mark one provider's PENDING rows of a run FAILED in a single UPDATE.

    The PENDING guard keeps a late stale/cleanup pass from overwriting a row a
    worker has just finished.

    Returns:
        int: Number of rows marked FAILED
    """
    statement = (
        update(TTSResult)
        .where(
            col(TTSResult.evaluation_run_id) == run_id,
            col(TTSResult.provider) == provider,
            col(TTSResult.status) == JobStatus.PENDING.value,
        )
        .values(
            status=JobStatus.FAILED.value,
            error_message=error_message,
            updated_at=now(),
        )
    )
    failed_count = session.exec(statement).rowcount
    session.commit()

    if failed_count:
        logger.warning(
            f"[fail_pending_tts_results] Failed pending results | run_id={run_id}, "
            f"provider={provider}, count={failed_count}, error={error_message}"
        )

    return failed_count


def save_processed_tts_results(
    *,
    session: Session,
    results: list[TTSResult],
) -> int:
    """Write rows finished in memory back to the DB, skipping any no longer PENDING.

    The cron re-dispatches result processing while a batch still has PENDING rows,
    so two tasks can process the same batch at once; the PENDING guard stops the
    later writer from overwriting a row the other already finished.

    Returns:
        int: Number of rows written
    """
    written = 0
    for result in results:
        statement = (
            update(TTSResult)
            .where(
                col(TTSResult.id) == result.id,
                col(TTSResult.status) == JobStatus.PENDING.value,
            )
            .values(
                {
                    TTSResult.status: result.status,
                    TTSResult.object_store_url: result.object_store_url,
                    TTSResult.metadata_: result.metadata_,
                    TTSResult.error_message: result.error_message,
                    col(TTSResult.usage): result.usage,
                    TTSResult.updated_at: result.updated_at,
                }
            )
        )
        written += session.exec(statement).rowcount
    session.commit()

    skipped = len(results) - written
    if skipped:
        logger.warning(
            f"[save_processed_tts_results] Skipped rows already finished by another "
            f"writer | count={skipped}"
        )

    return written


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
    the value; omitting it leaves it unchanged. `score` is merged per key into
    the stored dict (a None value deletes that key) because the UI rates one
    metric per request; `issues` replaces the stored list.

    Args:
        session: Database session
        result_id: Result ID
        org_id: Organization ID
        project_id: Project ID
        **kwargs: Fields to update (is_correct, comment, score, issues)

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

    if "score" in kwargs:
        merged_score = dict(result.score or {})
        for key, value in (kwargs["score"] or {}).items():
            if value is None:
                merged_score.pop(key, None)
            else:
                merged_score[key] = value
        result.score = merged_score or None

    if "issues" in kwargs:
        result.issues = kwargs["issues"] or None

    result.updated_at = now()

    session.add(result)
    session.commit()
    session.refresh(result)

    logger.info(
        f"[update_tts_human_feedback] Human feedback updated | "
        f"result_id: {result_id}, is_correct: {kwargs.get('is_correct')}"
    )

    return result


def get_tts_usage_by_model(
    *,
    session: Session,
    run_id: int,
    org_id: int,
    project_id: int,
) -> dict[str, dict[str, Any]]:
    """Sum each model's usage over all of a run's results in one aggregate query.

    Returns:
        dict[str, dict[str, Any]]: Per model, `unit`, the summed fields of that
            unit and `results_with_usage`; models with no usage are absent
    """
    all_fields: list[str] = []
    for fields in USAGE_SUM_FIELDS_BY_UNIT.values():
        for field in fields:
            if field not in all_fields:
                all_fields.append(field)

    usage_column = col(TTSResult.usage)
    sum_columns = []
    for field in all_fields:
        sql_type = Float if field in FRACTIONAL_USAGE_FIELDS else BigInteger
        # SUM(bigint) yields numeric (Decimal), which JSON-serializes as a string.
        sum_columns.append(
            func.sum(usage_column[field].astext.cast(sql_type)).cast(sql_type)
        )

    # One shared expression: separately built `->>` keys bind as different
    # parameters, which Postgres won't match between SELECT and GROUP BY.
    unit_column = usage_column["unit"].astext
    statement = (
        select(
            TTSResult.provider,
            unit_column,
            func.count(),
            *sum_columns,
        )
        .where(
            TTSResult.evaluation_run_id == run_id,
            TTSResult.organization_id == org_id,
            TTSResult.project_id == project_id,
            unit_column.is_not(None),
        )
        .group_by(TTSResult.provider, unit_column)
    )

    usage_by_model: dict[str, dict[str, Any]] = {}
    for row in session.exec(statement).all():
        provider, unit, results_with_usage = row[0], row[1], row[2]
        sums = dict(zip(all_fields, row[3:]))
        model_usage: dict[str, Any] = {"unit": unit}
        for field in USAGE_SUM_FIELDS_BY_UNIT.get(unit, ()):
            model_usage[field] = sums[field]
        model_usage["results_with_usage"] = results_with_usage
        usage_by_model[provider] = model_usage

    return usage_by_model


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

    return dict(rows)
