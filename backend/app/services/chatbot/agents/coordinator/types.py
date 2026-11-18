from dataclasses import dataclass

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    intent: str | None = Field(
        default=None,
        description="Exactly one intent name from the list, or null when none clearly matches.",
    )
    clarifying_question: str | None = Field(
        default=None,
        description="Question to ask the user when intent is null; otherwise null.",
    )


@dataclass(frozen=True)
class CoordinatorDecision:
    """Exactly one of `target_node_id` / `clarifying_question` is set."""

    target_node_id: str | None
    clarifying_question: str | None
