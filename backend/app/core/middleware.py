import logging
import time

import sentry_sdk
from asgi_correlation_id import correlation_id
from fastapi import Request, Response
from opentelemetry import trace
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import settings
from app.core.logger import log_service_name

logger = logging.getLogger("http_request_logger")

SILENT_LOG_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        f"{settings.API_V1_STR}/utils/health",
    }
)

CRON_PATH_PREFIX: str = f"{settings.API_V1_STR}/cron/"

# Excluded from traces only; logs/metrics and spans inside the handler still emit.
TRACE_EXCLUDED_PATH_PREFIXES: frozenset[str] = frozenset({CRON_PATH_PREFIX})


class StripTrailingSlashMiddleware:
    """
    Rewrite '/foo/' to '/foo' before routing so both forms hit the same handler.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
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


def _emit_http_metrics(
    *,
    method: str,
    http_route: str,
    status: int,
    duration_ms: float,
    request_body_size: int = 0,
) -> None:
    """Emit HTTP traffic/latency/payload/error counters to Sentry. No-op if the SDK is inactive."""
    try:
        if not sentry_sdk.get_client().is_active():
            return
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
        sentry_sdk.metrics.distribution(
            "http.server.request.body.size",
            request_body_size,
            unit="byte",
            attributes=attrs,
        )
        if status >= 400:
            sentry_sdk.metrics.count("http.server.request.error", 1, attributes=attrs)
    except Exception:
        logger.debug("[_emit_http_metrics] Sentry metric emit failed")


async def http_request_logger(request: Request, call_next) -> Response:
    if request.url.path.startswith(CRON_PATH_PREFIX):
        with log_service_name(settings.CRON_SERVICE_NAME):
            return await _log_http_request(request, call_next)

    return await _log_http_request(request, call_next)


def _resolve_request_body_size(request: Request) -> int:
    """
    Read the request payload size from Content-Length.
    Returns 0 when the header is absent or malformed (e.g. chunked transfer).
    """
    try:
        return int(request.headers.get("content-length") or 0)
    except ValueError:
        return 0


async def _log_http_request(request: Request, call_next) -> Response:
    start_time = time.time()
    method = request.method
    raw_path = request.url.path
    # Health/utility paths excluded so they don't skew platform traffic metrics.
    metrics_enabled = raw_path not in SILENT_LOG_PATHS
    request_body_size = _resolve_request_body_size(request)

    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("http.request.method", method)
        span.set_attribute("http.request_method", method)
        span.set_attribute("http.method", method)
        span.set_attribute("http.request.body.size", request_body_size)

    if sentry_sdk.get_client().is_active():
        sentry_sdk.set_tag("http.method", method)
        sentry_sdk.set_tag("http.request.method", method)
        if request_id := correlation_id.get():
            sentry_sdk.set_tag("correlation_id", request_id)

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.time() - start_time) * 1000
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
        if metrics_enabled:
            _emit_http_metrics(
                method=method,
                http_route=http_route,
                status=status,
                duration_ms=duration_ms,
                request_body_size=request_body_size,
            )
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

    if sentry_sdk.get_client().is_active():
        sentry_sdk.set_tag("http.route", http_route)
        sentry_sdk.set_tag("http.status_code", str(status))
        sentry_sdk.set_tag("http.response.status_code", str(status))

    if metrics_enabled:
        logger.info(
            f"[_log_http_request] {method} {raw_path} - {status} [{duration_ms:.2f}ms] "
            f"| request_body_size: {request_body_size}B "
            f"| correlation_id: {correlation_id.get()}"
        )
        _emit_http_metrics(
            method=method,
            http_route=http_route,
            status=status,
            duration_ms=duration_ms,
            request_body_size=request_body_size,
        )

    return response
