import logging
from datetime import datetime, timedelta
from typing import Any

import sentry_sdk
from sqlalchemy import text
from sqlmodel import Session

from app.core.telemetry import record_stale_pending_jobs
from app.core.util import now
from app.crud.job.pending_monitor import (
    StaleGroup,
    StalePendingResult,
    count_stale_pending_collection_jobs,
    count_stale_pending_doc_transformation_jobs,
    count_stale_llm_pending_jobs,
)
from app.models.collection_job import CollectionJobStatus
from app.models.doc_transformation_job import TransformationStatus
from app.models.job import JobStatus

logger = logging.getLogger(__name__)

PENDING_JOB_MONITOR_INTERVAL_MINUTES = 5

PENDING_RECENT_GRACE_MINUTES = 4
LLM_PENDING_THRESHOLD_MINUTES = 30
COLLECTION_PENDING_THRESHOLD_MINUTES = 30
DOC_TRANSFORMATION_PENDING_THRESHOLD_MINUTES = 30
PENDING_JOB_QUERY_TIMEOUT_MS = 1000

PENDING_JOBS_EVENT_MESSAGE = "Some jobs have been pending for longer than expected"

TABLE_DOMAIN_LABEL = {
    "job": "LLM job",
    "collection_jobs": "Collection job",
    "doc_transformation_job": "Document job",
}


def _age_seconds(oldest_at: datetime | None, as_of: datetime) -> int | None:
    if oldest_at is None:
        return None
    return max(0, int((as_of - oldest_at).total_seconds()))


def _apply_statement_timeout(session: Session, timeout_ms: int) -> None:
    """Cap subsequent queries in this transaction to ``timeout_ms``.

    No-op on dialects that do not support PostgreSQL's statement_timeout.
    """
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect != "postgresql":
        return
    session.exec(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))


def _format_group(group: StaleGroup, as_of: datetime) -> dict[str, Any]:
    formatted: dict[str, Any] = {
        "stale_count": group["stale_count"],
        "oldest_age_seconds": _age_seconds(group.get("oldest_at"), as_of),
    }
    if "job_type" in group:
        formatted["job_type"] = group["job_type"]
    if "action_type" in group:
        formatted["action_type"] = group["action_type"]
    return formatted


def _build_summary(
    *,
    table: str,
    status: str,
    threshold_minutes: int,
    result: StalePendingResult,
    as_of: datetime,
) -> dict[str, Any]:
    return {
        "table": table,
        "status": status,
        "threshold_minutes": threshold_minutes,
        "stale_count": result["stale_count"],
        "oldest_age_seconds": _age_seconds(result["oldest_at"], as_of),
        "groups": [_format_group(group, as_of) for group in result["groups"]],
        "ids": list(result.get("ids", [])),
        "ids_truncated": bool(result.get("ids_truncated", False)),
    }


def _emit_monitoring_metrics(summary: dict[str, Any]) -> None:
    record_stale_pending_jobs(
        table=summary["table"],
        status=summary["status"],
        stale_count=summary["stale_count"],
        oldest_age_seconds=summary["oldest_age_seconds"],
    )

    for group in summary["groups"]:
        record_stale_pending_jobs(
            table=summary["table"],
            status=summary["status"],
            stale_count=group["stale_count"],
            oldest_age_seconds=group["oldest_age_seconds"],
            job_type=group.get("job_type"),
            action_type=group.get("action_type"),
            dimensional=True,
        )


def _group_label(group: dict[str, Any]) -> str:
    return str(group.get("job_type") or group.get("action_type") or "unknown")


def _format_breakdown(summary: dict[str, Any]) -> str:
    """Compact one-line breakdown, e.g. `llm_call=12(oldest=900s), eval=3(oldest=120s)`."""
    parts: list[str] = []
    for group in summary["groups"]:
        label = _group_label(group)
        count = group["stale_count"]
        oldest = group.get("oldest_age_seconds")
        oldest_str = f"(oldest={oldest}s)" if oldest is not None else ""
        parts.append(f"{label}={count}{oldest_str}")
    return ", ".join(parts) if parts else f"total={summary['stale_count']}"


def _capture_stale_pending_jobs_event(summary: dict[str, Any]) -> None:
    if summary["stale_count"] == 0:
        return

    table = summary["table"]
    status = summary["status"]
    breakdown = _format_breakdown(summary)
    domain = TABLE_DOMAIN_LABEL.get(table, table)
    stale_count = summary["stale_count"]
    oldest = summary.get("oldest_age_seconds")
    threshold = summary["threshold_minutes"]
    title = f"[TEST]: {domain} pending for longer than expected"
    oldest_str = f"{oldest}s" if oldest is not None else "n/a"
    detail = (
        f"{title} — {stale_count} {domain} row(s) {status} inside stale window "
        f"[{PENDING_RECENT_GRACE_MINUTES}min, {threshold}min] "
        f"(oldest={oldest_str}). breakdown: {breakdown}"
    )

    tags: dict[str, Any] = {
        "monitor": "pending-jobs",
        "job.domain": domain,
        "job.table": table,
        "job.status": status,
        "job.stale_count": summary["stale_count"],
        "job.oldest_age_seconds": summary["oldest_age_seconds"],
        "job.breakdown": breakdown,
        "job.ids_truncated": summary.get("ids_truncated", False),
    }
    for group in summary["groups"]:
        label = _group_label(group)
        tags[f"pending.{table}.{label}"] = group["stale_count"]

    ids = summary.get("ids", [])
    try:
        sentry_sdk.capture_event(
            {
                "message": title,
                "logentry": {"formatted": detail},
                "level": "warning",
                "fingerprint": ["pending-jobs", table, status],
                "tags": tags,
                "extra": {
                    "pending_breakdown": breakdown,
                    "pending_groups": summary["groups"],
                    "pending_ids": ids,
                    "pending_ids_truncated": summary.get("ids_truncated", False),
                    "pending_ids_shown": len(ids),
                    "pending_ids_total": summary["stale_count"],
                },
                "contexts": {
                    "pending_jobs": {
                        "table": table,
                        "status": status,
                        "stale_count": summary["stale_count"],
                        "oldest_age_seconds": summary["oldest_age_seconds"],
                        "threshold_minutes": summary["threshold_minutes"],
                        "breakdown": breakdown,
                        "groups": summary["groups"],
                        "ids": ids,
                        "ids_shown": len(ids),
                        "ids_truncated": summary.get("ids_truncated", False),
                    }
                },
            }
        )
    except Exception:
        logger.warning(
            "[job_monitoring] Failed to capture Sentry event for table=%s",
            table,
            exc_info=True,
        )


def monitor_pending_jobs(session: Session) -> dict[str, Any]:
    """
    Check for jobs that are still PENDING after the expected queue pickup window.

    Emits counts, oldest age, and a capped oldest-first ID sample per table.
    DB-layer queries delegated to ``app.crud.job.pending_monitor``.
    """
    as_of = now()
    _apply_statement_timeout(session, PENDING_JOB_QUERY_TIMEOUT_MS)

    upper_cutoff = as_of - timedelta(minutes=PENDING_RECENT_GRACE_MINUTES)

    summaries = [
        _build_summary(
            table="job",
            status=JobStatus.PENDING.value,
            threshold_minutes=LLM_PENDING_THRESHOLD_MINUTES,
            result=count_stale_llm_pending_jobs(
                session,
                lower_cutoff=as_of - timedelta(minutes=LLM_PENDING_THRESHOLD_MINUTES),
                upper_cutoff=upper_cutoff,
            ),
            as_of=as_of,
        ),
        _build_summary(
            table="collection_jobs",
            status=CollectionJobStatus.PENDING.value,
            threshold_minutes=COLLECTION_PENDING_THRESHOLD_MINUTES,
            result=count_stale_pending_collection_jobs(
                session,
                lower_cutoff=as_of
                - timedelta(minutes=COLLECTION_PENDING_THRESHOLD_MINUTES),
                upper_cutoff=upper_cutoff,
            ),
            as_of=as_of,
        ),
        _build_summary(
            table="doc_transformation_job",
            status=TransformationStatus.PENDING.value,
            threshold_minutes=DOC_TRANSFORMATION_PENDING_THRESHOLD_MINUTES,
            result=count_stale_pending_doc_transformation_jobs(
                session,
                lower_cutoff=as_of
                - timedelta(minutes=DOC_TRANSFORMATION_PENDING_THRESHOLD_MINUTES),
                upper_cutoff=upper_cutoff,
            ),
            as_of=as_of,
        ),
    ]

    for summary in summaries:
        _emit_monitoring_metrics(summary)
        _capture_stale_pending_jobs_event(summary)

    result = {
        "status": "ok",
        "checked_at": as_of.isoformat(),
        "total_stale_pending": sum(summary["stale_count"] for summary in summaries),
        "tables": summaries,
    }

    logger.info(
        "[monitor_pending_jobs] Completed pending job monitor | stale_count=%s",
        result["total_stale_pending"],
    )
    return result


__all__ = [
    "COLLECTION_PENDING_THRESHOLD_MINUTES",
    "DOC_TRANSFORMATION_PENDING_THRESHOLD_MINUTES",
    "LLM_PENDING_THRESHOLD_MINUTES",
    "PENDING_JOB_MONITOR_INTERVAL_MINUTES",
    "PENDING_RECENT_GRACE_MINUTES",
    "monitor_pending_jobs",
]
