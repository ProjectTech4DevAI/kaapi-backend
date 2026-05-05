import logging
import time

import sentry_sdk
from fastapi import Request, Response
from opentelemetry import trace

logger = logging.getLogger("http_request_logger")


class StripTrailingSlashMiddleware:
    """
    Rewrite '/foo/' to '/foo' before routing so both forms hit the same handler.

    Why: removing trailing slashes from declared routes would break clients
    that don't follow 307 redirects on POST/PUT/DELETE or that drop the
    Authorization header across redirects. This middleware preserves the
    trailing-slash form during the deprecation window — to be removed once
    integrators have migrated.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            if len(path) > 1 and path.endswith("/"):
                scope = dict(scope)
                scope["path"] = path[:-1]
                raw_path = scope.get("raw_path")
                if raw_path is not None and raw_path.endswith(b"/"):
                    scope["raw_path"] = raw_path[:-1]
        await self.app(scope, receive, send)


def _resolve_http_route(request: Request) -> str:
    """
    Resolve the HTTP route for telemetry and logging.
    Uses the route's path template if available, otherwise falls back to the raw path.
    """
    route = request.scope.get("route")
    templated = getattr(route, "path", None)
    return templated or "unmatched"


async def http_request_logger(request: Request, call_next) -> Response:
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
