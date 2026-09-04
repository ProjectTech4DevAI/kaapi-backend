import logging
import re

import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.core.middleware import _resolve_request_body_size


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
