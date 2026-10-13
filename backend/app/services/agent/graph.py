"""LangGraph tool-use loop for the read-only /agent endpoint.

START -> agent_node -> (tool_use? tool_executor_node -> agent_node : END).
Claude plans which read-only Kaapi routes to call; the harness decides what is
callable (see tools.py) and attaches credentials (see executor.py).
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any, cast

import anthropic
import httpx
from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ToolChoiceParam,
)
from asgi_correlation_id import correlation_id
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from starlette.types import ASGIApp

from app.core.config import settings
from app.models.agent import (
    AgentQueryResponse,
    AgentStopReasonEnum,
    AgentToolCallPublic,
    AgentUsagePublic,
)
from app.services.agent.exceptions import AgentLLMError, AgentNotConfiguredError
from app.services.agent.executor import REQUEST_ID_HEADER, execute_tool_call
from app.services.agent.prompts import (
    AGENT_EMPTY_ANSWER_MESSAGE,
    AGENT_REFUSAL_MESSAGE,
    AGENT_SYSTEM_PROMPT,
)
from app.services.agent.state import AgentContext, AgentState
from app.services.agent.tools import ANTHROPIC_TOOL_DEFINITIONS
from app.services.llm.providers.claude import (
    STOP_REASON_COMPLETE,
    describe_anthropic_error,
    log_anthropic_error,
)

logger = logging.getLogger(__name__)

AGENT_NODE = "agent_node"
TOOL_EXECUTOR_NODE = "tool_executor_node"

# Never resolved over the network: ASGITransport routes it straight into the app.
INTERNAL_BASE_URL = "http://kaapi-internal"

_STOP_REASON_TOOL_USE = "tool_use"
_STOP_REASON_MAX_TOKENS = "max_tokens"
_STOP_REASON_REFUSAL = "refusal"
_BLOCK_TYPE_TEXT = "text"
_BLOCK_TYPE_TOOL_USE = "tool_use"

_SYSTEM_BLOCKS: list[TextBlockParam] = [
    {
        "type": "text",
        "text": AGENT_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


def _resolve_stop_reason(
    anthropic_stop_reason: str | None, tools_enabled: bool
) -> AgentStopReasonEnum:
    if anthropic_stop_reason == _STOP_REASON_MAX_TOKENS:
        return AgentStopReasonEnum.MAX_TOKENS
    if anthropic_stop_reason == _STOP_REASON_REFUSAL:
        return AgentStopReasonEnum.REFUSAL
    if not tools_enabled:
        return AgentStopReasonEnum.ITERATION_LIMIT
    return AgentStopReasonEnum.COMPLETED


async def agent_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict[str, Any]:
    tools_enabled = state["iterations"] < settings.AGENT_MAX_ITERATIONS
    tool_choice: ToolChoiceParam = (
        {"type": "auto"} if tools_enabled else {"type": "none"}
    )

    try:
        response = await runtime.context.llm_client.messages.create(
            model=settings.AGENT_MODEL,
            max_tokens=settings.AGENT_MAX_TOKENS,
            system=_SYSTEM_BLOCKS,
            tools=ANTHROPIC_TOOL_DEFINITIONS,
            tool_choice=tool_choice,
            messages=cast(list[MessageParam], state["messages"]),
            # AGENT_EFFORT is a plain str setting; the API rejects unknown values.
            output_config=cast(OutputConfigParam, {"effort": settings.AGENT_EFFORT}),
        )
    except anthropic.APIError as exc:
        log_anthropic_error(
            exc,
            fn_name="agent_node",
            context=f"model={settings.AGENT_MODEL}, iteration={state['iterations']}",
        )
        raise AgentLLMError(describe_anthropic_error(exc)) from exc

    usage = response.usage
    # Cached prefix tokens are billed separately from input_tokens; count all of them.
    input_tokens = (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )

    assistant_content: list[dict[str, Any]] = []
    has_tool_use = False
    text_parts: list[str] = []
    for block in response.content:
        assistant_content.append(block.to_dict(mode="json"))
        if block.type == _BLOCK_TYPE_TOOL_USE:
            has_tool_use = True
        elif block.type == _BLOCK_TYPE_TEXT:
            text_parts.append(block.text)

    update: dict[str, Any] = {
        "messages": [{"role": "assistant", "content": assistant_content}],
        "input_tokens": input_tokens,
        "output_tokens": usage.output_tokens,
    }

    if response.stop_reason == _STOP_REASON_TOOL_USE and tools_enabled and has_tool_use:
        return update

    stop_reason = _resolve_stop_reason(response.stop_reason, tools_enabled)
    answer = "\n\n".join(text_parts).strip()
    if not answer:
        answer = (
            AGENT_REFUSAL_MESSAGE
            if stop_reason == AgentStopReasonEnum.REFUSAL
            else AGENT_EMPTY_ANSWER_MESSAGE
        )
    if response.stop_reason not in (STOP_REASON_COMPLETE, _STOP_REASON_TOOL_USE):
        logger.warning(
            f"[agent_node] Model stopped early | stop_reason: {response.stop_reason}, "
            f"iteration: {state['iterations']}"
        )

    update["answer"] = answer
    update["stop_reason"] = stop_reason.value
    return update


def route_after_agent(state: AgentState) -> str:
    if state["answer"] is None:
        return TOOL_EXECUTOR_NODE
    return END


async def tool_executor_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict[str, Any]:
    last_message = state["messages"][-1]

    tool_use_blocks: list[dict[str, Any]] = []
    for block in last_message["content"]:
        if block.get("type") == _BLOCK_TYPE_TOOL_USE:
            tool_use_blocks.append(block)

    outcomes = await asyncio.gather(
        *(
            execute_tool_call(
                http_client=runtime.context.http_client,
                forwarded_headers=runtime.context.forwarded_headers,
                name=block["name"],
                raw_args=block.get("input") or {},
            )
            for block in tool_use_blocks
        )
    )

    tool_results: list[dict[str, Any]] = []
    tool_call_logs: list[dict[str, Any]] = []
    for block, outcome in zip(tool_use_blocks, outcomes, strict=True):
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": outcome.content,
                "is_error": outcome.is_error,
            }
        )
        tool_call_logs.append(
            {
                "name": block["name"],
                "arguments": outcome.arguments,
                "status_code": outcome.status_code,
                "is_error": outcome.is_error,
                "duration_ms": outcome.duration_ms,
            }
        )

    return {
        "messages": [{"role": "user", "content": tool_results}],
        "tool_calls": tool_call_logs,
        "iterations": state["iterations"] + 1,
    }


@lru_cache(maxsize=1)
def build_agent_graph() -> (
    CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]
):
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node(AGENT_NODE, agent_node)
    graph.add_node(TOOL_EXECUTOR_NODE, tool_executor_node)

    graph.add_edge(START, AGENT_NODE)
    graph.add_conditional_edges(
        AGENT_NODE, route_after_agent, [TOOL_EXECUTOR_NODE, END]
    )
    graph.add_edge(TOOL_EXECUTOR_NODE, AGENT_NODE)

    # No checkpointer in v1: every query is a single self-contained turn.
    return graph.compile()


async def run_agent_query(
    *,
    query: str,
    forwarded_headers: dict[str, str],
    asgi_app: ASGIApp,
) -> AgentQueryResponse:
    if not settings.ANTHROPIC_API_KEY:
        raise AgentNotConfiguredError(
            "[KAAPI] The agent is not configured on this server (missing platform "
            "Anthropic key). Contact Kaapi."
        )

    headers = dict(forwarded_headers)
    request_id = correlation_id.get()
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": query}],
        "tool_calls": [],
        "iterations": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "answer": None,
        "stop_reason": None,
    }
    # Each tool round is two supersteps (agent + executor), plus the final answer turn.
    recursion_limit = 2 * settings.AGENT_MAX_ITERATIONS + 5

    transport = httpx.ASGITransport(app=asgi_app, raise_app_exceptions=False)
    async with (
        httpx.AsyncClient(
            transport=transport,
            base_url=INTERNAL_BASE_URL,
            follow_redirects=False,
            timeout=settings.AGENT_TOOL_TIMEOUT_SECONDS,
        ) as http_client,
        AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.AGENT_LLM_TIMEOUT_SECONDS,
        ) as llm_client,
    ):
        final_state = await build_agent_graph().ainvoke(
            initial_state,
            config={"recursion_limit": recursion_limit},
            context=AgentContext(
                forwarded_headers=headers,
                http_client=http_client,
                llm_client=llm_client,
            ),
        )

    tool_calls: list[AgentToolCallPublic] = []
    for tool_call in final_state["tool_calls"]:
        tool_calls.append(AgentToolCallPublic.model_validate(tool_call))

    return AgentQueryResponse(
        answer=final_state["answer"] or AGENT_EMPTY_ANSWER_MESSAGE,
        stop_reason=AgentStopReasonEnum(final_state["stop_reason"]),
        iterations=final_state["iterations"],
        model=settings.AGENT_MODEL,
        tool_calls=tool_calls,
        usage=AgentUsagePublic(
            input_tokens=final_state["input_tokens"],
            output_tokens=final_state["output_tokens"],
        ),
    )
