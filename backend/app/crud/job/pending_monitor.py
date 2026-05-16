"""CRUD queries that back the pending-job monitor.

Returns aggregate counts plus a capped list of pending IDs (for Sentry visibility).
Used by ``app/services/job_monitoring.py``.
"""

from datetime import datetime
from typing import Any, TypedDict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlmodel import Session, select

from app.models.collection_job import CollectionJob, CollectionJobStatus
from app.models.doc_transformation_job import (
    DocTransformationJob,
    TransformationStatus,
)
from app.models.job import Job, JobStatus

MAX_IDS = 100


class StaleGroup(TypedDict, total=False):
    job_type: str
    action_type: str
    stale_count: int
    oldest_at: datetime | None


class StalePendingResult(TypedDict):
    stale_count: int
    oldest_at: datetime | None
    groups: list[StaleGroup]
    ids: list[str]
    ids_truncated: bool


def _count_and_oldest(
    session: Session,
    model: type[Any],
    timestamp_column: InstrumentedAttribute,
    conditions: tuple[Any, ...],
) -> tuple[int, datetime | None]:
    statement = select(func.count(model.id), func.min(timestamp_column)).where(
        *conditions
    )
    count, oldest_at = session.exec(statement).one()
    return int(count or 0), oldest_at


def _sample_ids(
    session: Session,
    model: type[Any],
    timestamp_column: InstrumentedAttribute,
    conditions: tuple[Any, ...],
    cap: int = MAX_IDS,
) -> list[str]:
    """Oldest-first ID sample, capped. Oldest first = most actionable."""
    statement = (
        select(model.id).where(*conditions).order_by(timestamp_column.asc()).limit(cap)
    )
    rows = session.exec(statement).all()
    return [str(row) if isinstance(row, UUID) else str(row) for row in rows]


def count_stale_llm_pending_jobs(
    session: Session,
    *,
    lower_cutoff: datetime,
    upper_cutoff: datetime,
) -> StalePendingResult:
    """Count Job rows pending inside [lower_cutoff, upper_cutoff], grouped by job_type."""
    conditions = (
        Job.status == JobStatus.PENDING,
        Job.inserted_at <= upper_cutoff,
        Job.inserted_at >= lower_cutoff,
    )
    stale_count, oldest_at = _count_and_oldest(
        session, Job, Job.inserted_at, conditions
    )

    group_statement = (
        select(Job.job_type, func.count(Job.id), func.min(Job.inserted_at))
        .where(*conditions)
        .group_by(Job.job_type)
    )
    groups: list[StaleGroup] = [
        {
            "job_type": str(job_type.value if hasattr(job_type, "value") else job_type),
            "stale_count": int(count or 0),
            "oldest_at": group_oldest_at,
        }
        for job_type, count, group_oldest_at in session.exec(group_statement).all()
    ]

    ids = _sample_ids(session, Job, Job.inserted_at, conditions)
    return {
        "stale_count": stale_count,
        "oldest_at": oldest_at,
        "groups": groups,
        "ids": ids,
        "ids_truncated": stale_count > len(ids),
    }


def count_stale_pending_collection_jobs(
    session: Session,
    *,
    lower_cutoff: datetime,
    upper_cutoff: datetime,
) -> StalePendingResult:
    """Count CollectionJob rows pending inside the window, grouped by action_type."""
    conditions = (
        CollectionJob.status == CollectionJobStatus.PENDING,
        CollectionJob.inserted_at <= upper_cutoff,
        CollectionJob.inserted_at >= lower_cutoff,
    )
    stale_count, oldest_at = _count_and_oldest(
        session, CollectionJob, CollectionJob.inserted_at, conditions
    )

    group_statement = (
        select(
            CollectionJob.action_type,
            func.count(CollectionJob.id),
            func.min(CollectionJob.inserted_at),
        )
        .where(*conditions)
        .group_by(CollectionJob.action_type)
    )
    groups: list[StaleGroup] = [
        {
            "action_type": str(
                action_type.value if hasattr(action_type, "value") else action_type
            ),
            "stale_count": int(count or 0),
            "oldest_at": group_oldest_at,
        }
        for action_type, count, group_oldest_at in session.exec(group_statement).all()
    ]

    ids = _sample_ids(session, CollectionJob, CollectionJob.inserted_at, conditions)
    return {
        "stale_count": stale_count,
        "oldest_at": oldest_at,
        "groups": groups,
        "ids": ids,
        "ids_truncated": stale_count > len(ids),
    }


def count_stale_pending_doc_transformation_jobs(
    session: Session,
    *,
    lower_cutoff: datetime,
    upper_cutoff: datetime,
) -> StalePendingResult:
    """Count DocTransformationJob rows pending inside the window."""
    conditions = (
        DocTransformationJob.status == TransformationStatus.PENDING,
        DocTransformationJob.inserted_at <= upper_cutoff,
        DocTransformationJob.inserted_at >= lower_cutoff,
    )
    stale_count, oldest_at = _count_and_oldest(
        session, DocTransformationJob, DocTransformationJob.inserted_at, conditions
    )
    ids = _sample_ids(
        session, DocTransformationJob, DocTransformationJob.inserted_at, conditions
    )
    return {
        "stale_count": stale_count,
        "oldest_at": oldest_at,
        "groups": [],
        "ids": ids,
        "ids_truncated": stale_count > len(ids),
    }


__all__ = [
    "MAX_IDS",
    "StaleGroup",
    "StalePendingResult",
    "count_stale_pending_collection_jobs",
    "count_stale_pending_doc_transformation_jobs",
    "count_stale_llm_pending_jobs",
]
