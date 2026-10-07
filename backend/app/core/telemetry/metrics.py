"""Sentry metrics emitter plus platform monitors: stale pending jobs, rate-threshold alerts."""

import logging
import resource
import sys
import threading
import time
from collections.abc import Mapping
from typing import Literal

import sentry_sdk
from opentelemetry import trace

logger = logging.getLogger(__name__)

MetricType = Literal["count", "gauge", "distribution"]
MetricAttributes = Mapping[str, str | int | float]
PROCESS_RSS_ATTRIBUTE = "process.memory.rss"
PROCESS_RSS_METRIC = "process.memory.rss"
_RSS_BYTES_PER_UNIT = 1 if sys.platform == "darwin" else 1024
_PROC_STATM = "/proc/self/statm"


def emit_sentry_metric(
    metric_type: MetricType,
    name: str,
    value: float,
    *,
    unit: str | None = None,
    attributes: MetricAttributes | None = None,
) -> None:
    """Best-effort Sentry metric emission. No-op if SDK is not active."""
    try:
        if not sentry_sdk.get_client().is_active():
            return
        emitter = getattr(sentry_sdk.metrics, metric_type)
        emitter(name=name, value=value, unit=unit, attributes=dict(attributes or {}))
    except Exception:
        logger.debug("[emit_sentry_metric] Failed to emit %s (%s)", name, metric_type)


def record_stale_pending_jobs(
    *,
    table: str,
    status: str,
    stale_count: int,
    oldest_age_seconds: int | None,
    job_type: str | None = None,
    action_type: str | None = None,
    dimensional: bool = False,
) -> None:
    """Emit aggregate pending-job monitor metrics to Sentry.

    When ``dimensional`` is True, the metric is emitted under a separate
    name so per-group counts (grouped by job_type/action_type) do not get
    summed together with the table-level count in Sentry dashboards.
    """
    attrs: dict[str, str] = {
        "job.table": table,
        "job.status": status,
    }
    if job_type:
        attrs["job.type"] = job_type
    if action_type:
        attrs["job.action_type"] = action_type

    count_metric = (
        "jobs.pending.stale.by_dimension.count"
        if dimensional
        else "jobs.pending.stale.count"
    )
    age_metric = (
        "jobs.pending.oldest_age_seconds.by_dimension"
        if dimensional
        else "jobs.pending.oldest_age_seconds"
    )

    emit_sentry_metric(
        "gauge",
        count_metric,
        stale_count,
        attributes=attrs,
    )
    if oldest_age_seconds is not None:
        emit_sentry_metric(
            "gauge",
            age_metric,
            oldest_age_seconds,
            unit="second",
            attributes=attrs,
        )


def record_rate_threshold(
    *,
    project_id: int,
    project_name: str | None,
    category: str,
    request_count: int,
    threshold: int,
) -> None:
    """Emit rate threshold exceeded event to Sentry."""

    try:
        if not sentry_sdk.get_client().is_active():
            return
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("alert.type", "threshold_rate_monitor")
            scope.set_tag("kaapi.project_id", project_id)
            scope.set_tag("route_category", category)
            scope.set_extra("request_count", request_count)
            scope.set_extra("threshold", threshold)
            sentry_sdk.capture_message(
                f"[Threshold-Monitor] {category} rate limit exceeded for project {project_id} | {project_name}: {request_count} req/min "
                f"(limit {threshold}/min)",
                level="warning",
            )
    except Exception as e:
        logger.exception("[record_rate_threshold] Failed to emit alert", exc_info=e)


def _current_rss_bytes() -> int:
    """Resident memory of this process right now; peak (ru_maxrss) is the fallback off Linux."""
    try:
        with open(_PROC_STATM) as statm:
            return int(statm.read().split()[1]) * resource.getpagesize()
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_BYTES_PER_UNIT


def record_process_rss(*, role: str, task_name: str | None = None) -> int:
    """Gauge `process.memory.rss` to Sentry and stamp it on the active span; returns bytes.

    Called around each Celery task so memory growth can be attributed to a task
    and tenant when a worker is recycled or killed for exceeding its memory limit.
    """
    rss = _current_rss_bytes()
    attrs: dict[str, str | int | float] = {"process.role": role}
    if task_name:
        attrs["celery.task_name"] = task_name
    emit_sentry_metric("gauge", PROCESS_RSS_METRIC, rss, unit="byte", attributes=attrs)
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(PROCESS_RSS_ATTRIBUTE, rss)
    return rss


def start_process_rss_monitor(*, role: str, interval_seconds: int) -> None:
    """Background thread gauging the worker parent process RSS every interval, since no task runs there."""

    def _loop() -> None:
        while True:
            record_process_rss(role=role)
            time.sleep(interval_seconds)

    threading.Thread(target=_loop, name="kaapi-rss-monitor", daemon=True).start()
