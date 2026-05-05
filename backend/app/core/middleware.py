import logging
import time

import sentry_sdk
from asgi_correlation_id import correlation_id
from fastapi import Request, Response
from opentelemetry import trace

from app.core.config import settings
from app.core.logger import log_service_name

logger = logging.getLogger("http_request_logger")


def _resolve_http_route(request: Request) -> str:
    """
    Resolve the HTTP route for telemetry and logging.
    Uses the route's path template if available, otherwise falls back to the raw path.
    """
    route = request.scope.get("route")
    templated = getattr(route, "path", None)
    return templated or "unmatched"


async def http_request_logger(request: Request, call_next) -> Response:
    if request.url.path.startswith(f"{settings.API_V1_STR}/cron/"):
        with log_service_name(settings.CRON_SERVICE_NAME):
            return await _log_http_request(request, call_next)

    return await _log_http_request(request, call_next)


async def _log_http_request(request: Request, call_next) -> Response:
    start_time = time.time()
    method = request.method
    raw_path = request.url.path

    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("http.request.method", method)
        span.set_attribute("http.request_method", method)
        span.set_attribute("http.method", method)

    if sentry_sdk.get_client().is_active():
        sentry_sdk.set_tag("http.method", method)
        sentry_sdk.set_tag("http.request.method", method)
        if request_id := correlation_id.get():
            sentry_sdk.set_tag("correlation_id", request_id)

    try:
        response = await call_next(request)
    except Exception:
        status = 500
        http_route = _resolve_http_route(request)
        if span.is_recording():
            span.set_attribute("http.route", http_route)
            span.set_attribute("http.status_code", status)
            span.set_attribute("http.response.status_code", status)
        if sentry_sdk.get_client().is_active():
            sentry_sdk.set_tag("http.route", http_route)
            sentry_sdk.set_tag("http.status_code", str(status))
            sentry_sdk.set_tag("http.response.status_code", str(status))
        logger.exception("Unhandled exception during request")
        raise

    duration_ms = (time.time() - start_time) * 1000
    status = response.status_code
    http_route = _resolve_http_route(request)

    if span.is_recording():
        span.set_attribute("http.route", http_route)
        span.set_attribute("http.status_code", status)
        span.set_attribute("http.response.status_code", status)
        span.set_attribute("http.request.duration_ms", round(duration_ms, 2))

    logger.info(f"{method} {raw_path} - {status} [{duration_ms:.2f}ms]")

    try:
        if sentry_sdk.get_client().is_active():
            sentry_sdk.set_tag("http.route", http_route)
            sentry_sdk.set_tag("http.status_code", str(status))
            sentry_sdk.set_tag("http.response.status_code", str(status))

            attrs = {
                "http.method": method,
                "http.route": http_route,
                "http.status_code": str(status),
            }
            sentry_sdk.metrics.count("http.server.request.count", 1, attributes=attrs)
            sentry_sdk.metrics.distribution(
                "http.server.request.duration",
                duration_ms,
                unit="millisecond",
                attributes=attrs,
            )
            if status >= 400:
                sentry_sdk.metrics.count(
                    "http.server.request.error", 1, attributes=attrs
                )
    except Exception:
        logger.debug("[http_request_logger] Sentry metric emit failed")

    return response
