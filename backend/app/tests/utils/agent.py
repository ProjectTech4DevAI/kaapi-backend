import copy
import json
from datetime import timedelta
from typing import Any

import pytest
from anthropic.types import Message

from app.core.config import settings
from app.core.security import create_access_token
from app.tests.utils.auth import TestAuthContext

ANTHROPIC_CLIENT_PATH = "app.services.agent.graph.AsyncAnthropic"

ScriptItem = Message | Exception


def anthropic_message(
    content: list[dict[str, Any]],
    stop_reason: str,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": settings.AGENT_MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


def text_message(text: str, stop_reason: str = "end_turn") -> Message:
    return anthropic_message([{"type": "text", "text": text}], stop_reason)


def tool_use_message(
    name: str, arguments: dict[str, Any], tool_use_id: str = "toolu_1"
) -> Message:
    return anthropic_message(
        [{"type": "tool_use", "id": tool_use_id, "name": name, "input": arguments}],
        "tool_use",
    )


class FakeAnthropic:
    """Stands in for `AsyncAnthropic`: replays a script and records every create() call.

    Once the script runs out, the last item is repeated so "always wants tools"
    models can be scripted with a single entry.
    """

    def __init__(self, script: list[ScriptItem]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    async def create(self, **kwargs: Any) -> Message:
        # Deep copy: graph state lists are appended to after the call returns.
        self.calls.append(copy.deepcopy(kwargs))
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, Exception):
            raise item
        return item

    async def __aenter__(self) -> "FakeAnthropic":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    def sent_payload(self) -> str:
        """Everything handed to the model across all calls, as one searchable string."""
        return json.dumps(self.calls, default=str)

    def tool_results(self, call_index: int) -> list[dict[str, Any]]:
        last_message = self.calls[call_index]["messages"][-1]
        return [
            block
            for block in last_message["content"]
            if block.get("type") == "tool_result"
        ]


def install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch, script: list[ScriptItem]
) -> FakeAnthropic:
    fake = FakeAnthropic(script)
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(ANTHROPIC_CLIENT_PATH, lambda **_kwargs: fake)
    return fake


def project_access_token(auth: TestAuthContext) -> str:
    return create_access_token(
        subject=str(auth.user_id),
        expires_delta=timedelta(minutes=30),
        organization_id=auth.organization_id,
        project_id=auth.project_id,
    )
