"""Assessment SSE event broadcaster."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class AssessmentEventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict]] = set()

    async def subscribe(self) -> AsyncIterator[str]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        self._subscribers.add(queue)
        logger.info("[subscribe] New SSE subscriber | total=%d", len(self._subscribers))
        try:
            yield "event: ready\ndata: {}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                event_type = payload.get("type", "message")
                yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        finally:
            self._subscribers.discard(queue)
            logger.info(
                "[subscribe] SSE subscriber disconnected | remaining=%d",
                len(self._subscribers),
            )

    def publish(self, payload: dict) -> None:
        if not self._subscribers:
            logger.debug(
                "[publish] No subscribers, event dropped | type=%s", payload.get("type")
            )
            return
        logger.info(
            "[publish] Broadcasting event | type=%s | subscribers=%d",
            payload.get("type"),
            len(self._subscribers),
        )
        for queue in list(self._subscribers):
            queue.put_nowait(payload)


assessment_event_broker = AssessmentEventBroker()
