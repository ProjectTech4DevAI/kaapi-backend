"""Sentry metrics emitter plus platform monitors: stale pending jobs, rate-threshold alerts."""

import logging
from collections.abc import Mapping
from typing import Literal

import sentry_sdk

from app.core.telemetry.constants import (
    ALERT_TYPE,
    HTTP_ROUTE_CATEGORY,
    JOB_ACTION_TYPE,
    JOB_STATUS,
    JOB_TABLE,
    JOB_TYPE,
    METRIC_JOB_PENDING_OLDEST_AGE,
    METRIC_JOB_PENDING_OLDEST_AGE_BY_DIMENSION,
    METRIC_JOB_PENDING_STALE_BY_DIMENSION_COUNT,
    METRIC_JOB_PENDING_STALE_COUNT,
    TENANT_PROJECT_ID,
)

logger = logging.getLogger(__name__)

MetricType = Literal["count", "gauge", "distribution"]
MetricAttributes = Mapping[str, str | int | float]


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
        JOB_TABLE: table,
        JOB_STATUS: status,
    }
    if job_type:
        attrs[JOB_TYPE] = job_type
    if action_type:
        attrs[JOB_ACTION_TYPE] = action_type

    count_metric = (
        METRIC_JOB_PENDING_STALE_BY_DIMENSION_COUNT
        if dimensional
        else METRIC_JOB_PENDING_STALE_COUNT
    )
    age_metric = (
        METRIC_JOB_PENDING_OLDEST_AGE_BY_DIMENSION
        if dimensional
        else METRIC_JOB_PENDING_OLDEST_AGE
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
        with sentry_sdk.new_scope() as scope:
            scope.set_tag(ALERT_TYPE, "threshold_rate_monitor")
            scope.set_tag(TENANT_PROJECT_ID, str(project_id))
            scope.set_tag(HTTP_ROUTE_CATEGORY, category)
            scope.set_extra("request_count", request_count)
            scope.set_extra("threshold", threshold)
            sentry_sdk.capture_message(
                f"[Threshold-Monitor] {category} rate limit exceeded for project {project_id} | {project_name}: {request_count} req/min "
                f"(limit {threshold}/min)",
                level="warning",
            )
    except Exception:
        logger.exception("[record_rate_threshold] Failed to emit alert")
