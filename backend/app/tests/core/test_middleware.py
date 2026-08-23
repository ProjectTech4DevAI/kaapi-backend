"""Tests for HTTP request metrics in core/middleware.py.

sentry_sdk and the OTel span are mocked; no real Sentry connection is used. The
middleware is driven directly with a fake Request + call_next so traffic, latency
and error counters can be asserted without an ASGI server.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import middleware


def _active_sentry() -> MagicMock:
    fake = MagicMock()
    fake.get_client.return_value.is_active.return_value = True
    return fake


def _request(path: str, method: str = "GET", route: str | None = None) -> MagicMock:
    req = MagicMock()
    req.url.path = path
    req.method = method
    req.scope = {"route": SimpleNamespace(path=route)} if route else {}
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
