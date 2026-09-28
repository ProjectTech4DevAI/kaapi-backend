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

    retryable: bool = False
    """Whether re-running the identical call could plausibly succeed. `error` is one
    string for every failure kind, so a retrying caller cannot classify it; this flag
    carries the classification instead. Defaults to False so a failure path added
    later fails fast rather than silently inheriting three attempts and its backoff."""

    @property
    def success(self) -> bool:
        return self.error is None and self.response is not None
