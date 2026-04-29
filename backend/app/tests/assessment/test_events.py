"""Tests for assessment/events.py."""

import asyncio

import pytest

from app.assessment.events import AssessmentEventBroker


class TestAssessmentEventBroker:
    def test_publish_without_subscribers_noop(self) -> None:
        broker = AssessmentEventBroker()
        broker.publish({"type": "x"})

    @pytest.mark.asyncio
    async def test_subscribe_receives_ready_and_event(self) -> None:
        broker = AssessmentEventBroker()
        stream = broker.subscribe()

        ready = await anext(stream)
        assert ready.startswith("event: ready")

        broker.publish({"type": "assessment.child_status_changed", "x": 1})
        event_chunk = await anext(stream)
        assert "event: assessment.child_status_changed" in event_chunk
        assert '"x": 1' in event_chunk

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_subscribe_keep_alive_on_timeout(self) -> None:
        broker = AssessmentEventBroker()
        stream = broker.subscribe()
        await anext(stream)  # ready

        original_wait_for = asyncio.wait_for

        async def fake_wait_for(coro, *args, **kwargs):  # type: ignore[no-untyped-def]
            coro.close()
            raise asyncio.TimeoutError

        try:
            asyncio.wait_for = fake_wait_for  # type: ignore[assignment]
            keep_alive = await anext(stream)
            assert keep_alive == ": keep-alive\n\n"
        finally:
            asyncio.wait_for = original_wait_for  # type: ignore[assignment]
            await stream.aclose()
