from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from app.models.llm.response import LLMCallResponse, Usage


class GuardrailOutcomeEnum(StrEnum):
    BLOCKED = "blocked"
    REPHRASED = "rephrased"


@dataclass
class BlockResult:
    """Result of a single block/LLM call execution."""

    response: LLMCallResponse | None = None
    llm_call_id: UUID | None = None
    usage: Usage | None = None
    error: str | None = None
    metadata: dict | None = None
    guardrail_outcome: GuardrailOutcomeEnum | None = None
    """Set when guardrails decided the outcome, vs. a provider failure."""

    retryable: bool = False
    """Whether retrying could succeed. False by default so new failures fail fast."""

    @property
    def success(self) -> bool:
        return self.error is None and self.response is not None
