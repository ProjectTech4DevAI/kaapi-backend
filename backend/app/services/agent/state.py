import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import httpx
from anthropic import AsyncAnthropic


class AgentState(TypedDict):
    """Graph state; kept JSON-serializable so a checkpointer can be added for multi-turn."""

    messages: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    iterations: int
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    answer: str | None
    stop_reason: str | None


@dataclass
class AgentContext:
    """Per-invocation runtime context; never persisted and never shown to the model."""

    forwarded_headers: dict[str, str]
    http_client: httpx.AsyncClient
    llm_client: AsyncAnthropic
