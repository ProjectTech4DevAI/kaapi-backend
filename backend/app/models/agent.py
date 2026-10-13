from enum import StrEnum
from typing import Any

from sqlmodel import Field, SQLModel

AGENT_QUERY_MAX_LENGTH = 4000


class AgentStopReasonEnum(StrEnum):
    COMPLETED = "completed"
    # Tool budget (AGENT_MAX_ITERATIONS) ran out, so the answer may be partial.
    ITERATION_LIMIT = "iteration_limit"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"


class AgentQueryRequest(SQLModel):
    query: str = Field(
        min_length=1,
        max_length=AGENT_QUERY_MAX_LENGTH,
        description=(
            "Natural-language question about your own project's data, e.g. "
            "'Summarize my last three evaluation runs.'"
        ),
    )


class AgentToolCallPublic(SQLModel):
    """One read-only Kaapi endpoint call the agent made while answering."""

    name: str = Field(description="Tool name from the agent's read-only registry")
    arguments: dict[str, Any] = Field(
        description="Arguments the model supplied (validated values when valid)"
    )
    status_code: int | None = Field(
        default=None,
        description="HTTP status of the internal request; null if no request was sent",
    )
    is_error: bool = Field(description="True if the call failed or was rejected")
    duration_ms: int = Field(description="Wall-clock time of the tool call")


class AgentUsagePublic(SQLModel):
    """Token usage summed across every model turn."""

    input_tokens: int
    output_tokens: int


class AgentQueryResponse(SQLModel):
    """Answer produced by the read-only agent."""

    answer: str
    stop_reason: AgentStopReasonEnum
    iterations: int = Field(description="Number of tool rounds executed")
    model: str
    tool_calls: list[AgentToolCallPublic]
    usage: AgentUsagePublic
