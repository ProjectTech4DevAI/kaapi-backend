import logging

import sentry_sdk

logger = logging.getLogger(__name__)


def _emit_sentry_metric(
    metric_type: str,
    name: str,
    value: float,
    *,
    unit: str | None = None,
    attributes: dict[str, str | int | float] | None = None,
) -> None:
    """Best-effort Sentry metric emission. No-op if SDK is not active."""
    try:
        if not sentry_sdk.get_client().is_active():
            return
        if metric_type == "count":
            sentry_sdk.metrics.count(
                name=name, value=value, unit=unit, attributes=attributes
            )
        elif metric_type == "gauge":
            sentry_sdk.metrics.gauge(
                name=name, value=value, unit=unit, attributes=attributes
            )
        elif metric_type == "distribution":
            sentry_sdk.metrics.distribution(
                name=name, value=value, unit=unit, attributes=attributes
            )
    except Exception:
        logger.debug("[_emit_sentry_metric] Failed to emit %s (%s)", name, metric_type)


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

    _emit_sentry_metric(
        "gauge",
        count_metric,
        stale_count,
        attributes=attrs,
    )
    if oldest_age_seconds is not None:
        _emit_sentry_metric(
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
            scope.set_tag("tenant.project_id", project_id)
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
