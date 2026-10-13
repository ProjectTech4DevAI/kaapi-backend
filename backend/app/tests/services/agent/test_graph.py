import json
from typing import Any

import pytest
from asgi_correlation_id import correlation_id
from fastapi import FastAPI, Request

from app.core.config import settings
from app.services.agent import AgentNotConfiguredError, run_agent_query
from app.tests.utils.agent import (
    anthropic_message,
    install_fake_anthropic,
    text_message,
    tool_use_message,
)


def _stub_kaapi_app(seen_headers: list[dict[str, str]]) -> FastAPI:
    stub = FastAPI()

    @stub.get(f"{settings.API_V1_STR}/collections")
    def list_collections(request: Request) -> dict[str, Any]:
        seen_headers.append(dict(request.headers))
        return {"success": True, "data": [{"id": "col-1", "name": "kb"}]}

    @stub.get(f"{settings.API_V1_STR}/evaluations/{{evaluation_id}}")
    def get_run(evaluation_id: int) -> dict[str, Any]:
        return {"success": True, "data": {"id": evaluation_id, "status": "completed"}}

    return stub


@pytest.mark.asyncio
class TestRunAgentQuery:
    async def test_assistant_turn_replayed_verbatim_including_thinking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thinking_block = {
            "type": "thinking",
            "thinking": "I should list collections.",
            "signature": "sig-abc",
        }
        tool_block = {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "list_collections",
            "input": {},
        }
        fake = install_fake_anthropic(
            monkeypatch,
            [
                anthropic_message([thinking_block, tool_block], "tool_use"),
                text_message("One collection: col-1."),
            ],
        )

        result = await run_agent_query(
            query="What collections do I have?",
            forwarded_headers={"X-API-KEY": "ApiKey k"},
            asgi_app=_stub_kaapi_app([]),
        )

        assert result.answer == "One collection: col-1."
        user_turn, assistant_turn, tool_turn = fake.calls[1]["messages"]
        assert user_turn == {"role": "user", "content": "What collections do I have?"}
        assert assistant_turn["role"] == "assistant"
        assert assistant_turn["content"] == [thinking_block, tool_block]
        assert tool_turn["role"] == "user"

    async def test_request_id_and_credentials_forwarded_to_inner_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_headers: list[dict[str, str]] = []
        install_fake_anthropic(
            monkeypatch,
            [tool_use_message("list_collections", {}), text_message("ok")],
        )
        token = correlation_id.set("corr-123")
        try:
            await run_agent_query(
                query="q",
                forwarded_headers={"Authorization": "Bearer tok"},
                asgi_app=_stub_kaapi_app(seen_headers),
            )
        finally:
            correlation_id.reset(token)

        [inner] = seen_headers
        assert inner["x-request-id"] == "corr-123"
        assert inner["authorization"] == "Bearer tok"
        assert "x-api-key" not in inner

    async def test_parallel_tool_results_keep_request_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = install_fake_anthropic(
            monkeypatch,
            [
                anthropic_message(
                    [
                        {
                            "type": "tool_use",
                            "id": "toolu_run",
                            "name": "get_evaluation_run",
                            "input": {"evaluation_id": 5},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_bad",
                            "name": "no_such_tool",
                            "input": {},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_cols",
                            "name": "list_collections",
                            "input": {},
                        },
                    ],
                    "tool_use",
                ),
                text_message("done"),
            ],
        )

        result = await run_agent_query(
            query="q", forwarded_headers={}, asgi_app=_stub_kaapi_app([])
        )

        results = fake.tool_results(1)
        assert [r["tool_use_id"] for r in results] == [
            "toolu_run",
            "toolu_bad",
            "toolu_cols",
        ]
        assert [r["is_error"] for r in results] == [False, True, False]
        assert json.loads(results[0]["content"]) == {"id": 5, "status": "completed"}
        assert [c.name for c in result.tool_calls] == [
            "get_evaluation_run",
            "no_such_tool",
            "list_collections",
        ]
        assert result.tool_calls[1].status_code is None
        assert result.iterations == 1

    async def test_usage_sums_all_turns_including_cache_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = tool_use_message("list_collections", {})
        first.usage.input_tokens = 100
        first.usage.cache_creation_input_tokens = 1000
        first.usage.output_tokens = 7
        second = text_message("ok")
        second.usage.input_tokens = 50
        second.usage.cache_read_input_tokens = 1000
        second.usage.output_tokens = 3
        install_fake_anthropic(monkeypatch, [first, second])

        result = await run_agent_query(
            query="q", forwarded_headers={}, asgi_app=_stub_kaapi_app([])
        )

        assert result.usage.input_tokens == 2150
        assert result.usage.output_tokens == 10

    async def test_system_prompt_is_cacheable_and_effort_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = install_fake_anthropic(monkeypatch, [text_message("hi")])

        await run_agent_query(
            query="q", forwarded_headers={}, asgi_app=_stub_kaapi_app([])
        )

        [call] = fake.calls
        assert call["model"] == settings.AGENT_MODEL
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert call["output_config"] == {"effort": settings.AGENT_EFFORT}
        assert call["tool_choice"] == {"type": "auto"}

    async def test_missing_key_raises_before_calling_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = install_fake_anthropic(monkeypatch, [text_message("unused")])
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

        with pytest.raises(AgentNotConfiguredError):
            await run_agent_query(
                query="q", forwarded_headers={}, asgi_app=_stub_kaapi_app([])
            )

        assert fake.calls == []
