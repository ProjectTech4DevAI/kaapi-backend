"""HTTP request metrics (count, duration, body size, errors) emitted from core/middleware.py."""

from app.core.telemetry.constants import (
    HTTP_METHOD,
    HTTP_ROUTE,
    HTTP_STATUS_CODE,
    METRIC_HTTP_REQUEST_BODY_SIZE,
    METRIC_HTTP_REQUEST_COUNT,
    METRIC_HTTP_REQUEST_DURATION,
    METRIC_HTTP_REQUEST_ERROR,
    METRIC_HTTP_REQUEST_UNMATCHED,
)
from app.core.telemetry.metrics import emit_sentry_metric

HTTP_ERROR_STATUS_THRESHOLD: int = 400
UNMATCHED_ROUTE: str = "unmatched"


def record_http_request(
    *,
    method: str,
    http_route: str,
    status: int,
    duration_ms: float,
    request_body_size: int = 0,
) -> None:
    """Emit HTTP traffic, latency, payload-size and error counters to Sentry."""
    attrs: dict[str, str | int | float] = {
        HTTP_METHOD: method,
        HTTP_ROUTE: http_route,
        HTTP_STATUS_CODE: str(status),
    }
    emit_sentry_metric("count", METRIC_HTTP_REQUEST_COUNT, 1, attributes=attrs)
    emit_sentry_metric(
        "distribution",
        METRIC_HTTP_REQUEST_DURATION,
        duration_ms,
        unit="millisecond",
        attributes=attrs,
    )
    emit_sentry_metric(
        "distribution",
        METRIC_HTTP_REQUEST_BODY_SIZE,
        request_body_size,
        unit="byte",
        attributes=attrs,
    )
    if status >= HTTP_ERROR_STATUS_THRESHOLD:
        emit_sentry_metric("count", METRIC_HTTP_REQUEST_ERROR, 1, attributes=attrs)


def record_unmatched_request(*, method: str) -> None:
    """Count requests that matched no route (scanners/bots) without a route label."""
    emit_sentry_metric(
        "count",
        METRIC_HTTP_REQUEST_UNMATCHED,
        1,
        attributes={HTTP_METHOD: method},
    )
