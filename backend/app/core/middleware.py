import logging
import time

import sentry_sdk
from fastapi import Request, Response
from opentelemetry import trace

logger = logging.getLogger("http_request_logger")


async def http_request_logger(request: Request, call_next) -> Response:
    start_time = time.time()
    method = request.method
    route = request.url.path

    # Set request-level dimensions as early as possible.
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("http.request.method", method)
        span.set_attribute("http.request_method", method)
        span.set_attribute("http.method", method)
        span.set_attribute("http.route", route)

    if sentry_sdk.get_client().is_active():
        sentry_sdk.set_tag("http.method", method)
        sentry_sdk.set_tag("http.request.method", method)
        sentry_sdk.set_tag("http.route", route)

    try:
        response = await call_next(request)
    except Exception:
        # Capture status for failed requests too so dashboards stay consistent.
        status = 500
        if span.is_recording():
            span.set_attribute("http.status_code", status)
            span.set_attribute("http.response.status_code", status)
        if sentry_sdk.get_client().is_active():
            sentry_sdk.set_tag("http.status_code", str(status))
            sentry_sdk.set_tag("http.response.status_code", str(status))
        logger.exception("Unhandled exception during request")
        raise

    duration_ms = (time.time() - start_time) * 1000
    status = response.status_code

    if span.is_recording():
        span.set_attribute("http.status_code", status)
        span.set_attribute("http.response.status_code", status)
        span.set_attribute("http.request.duration_ms", round(duration_ms, 2))

    logger.info(f"{method} {route} - {status} [{duration_ms:.2f}ms]")

    try:
        if sentry_sdk.get_client().is_active():
            sentry_sdk.set_tag("http.status_code", str(status))
            sentry_sdk.set_tag("http.response.status_code", str(status))

            attrs = {
                "http.method": method,
                "http.route": route,
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
