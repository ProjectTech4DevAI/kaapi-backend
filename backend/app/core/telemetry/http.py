"""HTTP request metrics (count, duration, body size, errors) emitted from core/middleware.py."""

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
        "http.method": method,
        "http.route": http_route,
        "http.status_code": str(status),
    }
    emit_sentry_metric("count", "http.server.request.count", 1, attributes=attrs)
    emit_sentry_metric(
        "distribution",
        "http.server.request.duration",
        duration_ms,
        unit="millisecond",
        attributes=attrs,
    )
    emit_sentry_metric(
        "distribution",
        "http.server.request.body.size",
        request_body_size,
        unit="byte",
        attributes=attrs,
    )
    if status >= HTTP_ERROR_STATUS_THRESHOLD:
        emit_sentry_metric("count", "http.server.request.error", 1, attributes=attrs)


def record_unmatched_request(*, method: str) -> None:
    """Count requests that matched no route (scanners/bots) without a route label."""
    emit_sentry_metric(
        "count",
        "http.server.request.unmatched",
        1,
        attributes={"http.method": method},
    )
