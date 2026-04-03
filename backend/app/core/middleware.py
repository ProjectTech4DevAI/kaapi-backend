import logging
import time
from fastapi import Request, Response
from opentelemetry import trace

logger = logging.getLogger("http")


async def http_request_logger(request: Request, call_next) -> Response:
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception during request")
        raise

    duration_ms = (time.time() - start_time) * 1000
    status = response.status_code

    # Enrich the active OTEL span with request-level attributes
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("http.route", request.url.path)
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.status_code", status)
        span.set_attribute("http.duration_ms", round(duration_ms, 2))
        span.set_attribute("http.client_ip", request.client.host if request.client else "unknown")

    # Console log: concise one-liner per request
    logger.info(
        "%s %s %d %.0fms",
        request.method,
        request.url.path,
        status,
        duration_ms,
    )
    return response
