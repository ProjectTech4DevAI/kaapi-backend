from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from app.models.llm.response import LLMCallResponse, Usage

GuardrailOutcomeLabel = Literal["blocked", "rephrased"]


@dataclass
class BlockResult:
    """Result of a single block/LLM call execution."""

    response: LLMCallResponse | None = None
    llm_call_id: UUID | None = None
    usage: Usage | None = None
    error: str | None = None
    metadata: dict | None = None
    guardrail_outcome: GuardrailOutcomeLabel | None = None
    """Set when guardrails decided the outcome, so a caller can tell a content
    verdict apart from a provider failure carried in `error`."""

    @property
    def success(self) -> bool:
        return self.error is None and self.response is not None
