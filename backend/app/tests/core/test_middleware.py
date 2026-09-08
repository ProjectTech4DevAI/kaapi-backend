"""Tests for HTTP request metrics and access logging in core/middleware.py.

sentry_sdk and the OTel span are mocked; no real Sentry connection is used. The
middleware is driven directly with a fake Request + call_next so traffic, latency,
payload-size and error counters can be asserted without an ASGI server.
"""

import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.core import middleware
from app.core.middleware import _resolve_request_body_size


def _active_sentry() -> MagicMock:
    fake = MagicMock()
    fake.get_client.return_value.is_active.return_value = True
    return fake


def _request(
    path: str,
    method: str = "GET",
    route: str | None = None,
    content_length: str | None = None,
) -> MagicMock:
    req = MagicMock()
    req.url.path = path
    req.method = method
    req.scope = {"route": SimpleNamespace(path=route)} if route else {}
    # Real dict: _resolve_request_body_size calls int() on whatever .get returns.
    req.headers = {"content-length": content_length} if content_length else {}
    return req


def _metric_names(fake: MagicMock) -> list[str]:
    calls = list(fake.metrics.count.call_args_list) + list(
        fake.metrics.distribution.call_args_list
    )
    return [c.args[0] for c in calls]


@pytest.fixture
def non_recording_span():
    span = MagicMock()
    span.is_recording.return_value = False
    with patch.object(middleware.trace, "get_current_span", return_value=span):
        yield span


class TestHttpRequestMetrics:
    async def _run(self, request, call_next, fake):
        with patch.object(middleware, "sentry_sdk", fake):
            return await middleware._log_http_request(request, call_next)

    @pytest.mark.asyncio
    async def test_success_emits_traffic_and_latency(self, non_recording_span):
        fake = _active_sentry()
        request = _request("/api/v1/items", route="/api/v1/items")
        call_next = AsyncMock(return_value=SimpleNamespace(status_code=200))

        await self._run(request, call_next, fake)

        names = _metric_names(fake)
        assert "http.server.request.count" in names
        assert "http.server.request.duration" in names
        assert "http.server.request.error" not in names

    @pytest.mark.asyncio
    async def test_client_error_status_emits_error_metric(self, non_recording_span):
        fake = _active_sentry()
        request = _request("/api/v1/items", route="/api/v1/items")
        call_next = AsyncMock(return_value=SimpleNamespace(status_code=404))

        await self._run(request, call_next, fake)

        error_calls = [
            c
            for c in fake.metrics.count.call_args_list
            if c.args[0] == "http.server.request.error"
        ]
        assert len(error_calls) == 1
        assert error_calls[0].kwargs["attributes"]["http.status_code"] == "404"

    @pytest.mark.asyncio
    async def test_unhandled_exception_counts_500_and_reraises(
        self, non_recording_span
    ):
        fake = _active_sentry()
        request = _request("/api/v1/items", route="/api/v1/items")
        call_next = AsyncMock(side_effect=ValueError("boom"))

        with pytest.raises(ValueError):
            await self._run(request, call_next, fake)

        error_calls = [
            c
            for c in fake.metrics.count.call_args_list
            if c.args[0] == "http.server.request.error"
        ]
        assert len(error_calls) == 1
        assert error_calls[0].kwargs["attributes"]["http.status_code"] == "500"

    @pytest.mark.asyncio
    async def test_health_path_excluded_from_metrics(self, non_recording_span):
        fake = _active_sentry()
        request = _request("/health", route="/health")
        call_next = AsyncMock(return_value=SimpleNamespace(status_code=200))

        await self._run(request, call_next, fake)

        fake.metrics.count.assert_not_called()
        fake.metrics.distribution.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_template_used_as_metric_tag(self, non_recording_span):
        fake = _active_sentry()
        request = _request("/api/v1/items/42", route="/api/v1/items/{item_id}")
        call_next = AsyncMock(return_value=SimpleNamespace(status_code=200))

        await self._run(request, call_next, fake)

        count_call = next(
            c
            for c in fake.metrics.count.call_args_list
            if c.args[0] == "http.server.request.count"
        )
        assert (
            count_call.kwargs["attributes"]["http.route"] == "/api/v1/items/{item_id}"
        )


def _build_request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/api/v1/probe",
            "raw_path": b"/api/v1/probe",
            "query_string": b"",
            "root_path": "",
            "headers": raw_headers,
        }
    )


class TestResolveRequestBodySize:
    def test_reads_content_length_header(self) -> None:
        request = _build_request({"content-length": "512"})
        assert _resolve_request_body_size(request) == 512

    def test_missing_header_is_zero(self) -> None:
        assert _resolve_request_body_size(_build_request()) == 0

    def test_empty_header_is_zero(self) -> None:
        request = _build_request({"content-length": ""})
        assert _resolve_request_body_size(request) == 0

    @pytest.mark.parametrize("value", ["chunked", "12.5", "-"])
    def test_malformed_header_is_zero(self, value: str) -> None:
        request = _build_request({"content-length": value})
        assert _resolve_request_body_size(request) == 0


class TestAccessLogLine:
    def test_logs_body_size_and_correlation_id(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = b"0123456789abcdef"

        with caplog.at_level(logging.INFO, logger="http_request_logger"):
            response = client.post("/api/v1/no-such-route", content=payload)

        assert response.status_code == 404
        line = next(
            record.getMessage()
            for record in caplog.records
            if record.name == "http_request_logger"
        )
        assert "POST /api/v1/no-such-route - 404" in line
        assert "| request_body_size: 16B |" in line
        assert re.search(r"correlation_id: [0-9a-f]{32}$", line)

    def test_logs_zero_for_bodyless_request(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="http_request_logger"):
            client.get("/api/v1/no-such-route")

        line = next(
            record.getMessage()
            for record in caplog.records
            if record.name == "http_request_logger"
        )
        assert "| request_body_size: 0B |" in line


class TestBodySizeSpanAttribute:
    @pytest.mark.asyncio
    async def test_recording_span_records_body_size(self) -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from app.core.middleware import _log_http_request

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer(__name__)

        async def call_next(_: Request) -> Response:
            return Response(status_code=200)

        request = _build_request({"content-length": "2048"})
        with trace.use_span(tracer.start_span("test-request"), end_on_exit=True):
            await _log_http_request(request, call_next)

        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["http.request.body.size"] == 2048


class TestSentryBodySizeMetric:
    @pytest.mark.asyncio
    async def test_emits_body_size_distribution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sentry_sdk

        from app.core.middleware import _log_http_request

        distributions: list[tuple[str, float, str | None]] = []

        class ActiveClient:
            def is_active(self) -> bool:
                return True

        monkeypatch.setattr(sentry_sdk, "get_client", lambda: ActiveClient())
        monkeypatch.setattr(
            sentry_sdk.metrics,
            "distribution",
            lambda name, value, unit=None, attributes=None: distributions.append(
                (name, value, unit)
            ),
        )
        monkeypatch.setattr(
            sentry_sdk.metrics, "count", lambda name, value, attributes=None: None
        )

        async def call_next(_: Request) -> Response:
            return Response(status_code=200)

        request = _build_request({"content-length": "4096"})
        await _log_http_request(request, call_next)

        body_size_metrics = [m for m in distributions if m[0].endswith("body.size")]
        assert body_size_metrics == [("http.server.request.body.size", 4096, "byte")]

    @pytest.mark.asyncio
    async def test_no_metrics_when_sentry_inactive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sentry_sdk

        from app.core.middleware import _log_http_request

        emitted: list[str] = []

        class InactiveClient:
            def is_active(self) -> bool:
                return False

        monkeypatch.setattr(sentry_sdk, "get_client", lambda: InactiveClient())
        monkeypatch.setattr(
            sentry_sdk.metrics,
            "distribution",
            lambda name, *a, **kw: emitted.append(name),
        )

        async def call_next(_: Request) -> Response:
            return Response(status_code=200)

        await _log_http_request(_build_request({"content-length": "77"}), call_next)

        assert emitted == []
