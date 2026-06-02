from dataclasses import dataclass
from uuid import UUID

from app.models.llm.response import LLMCallErrorDetail, LLMCallResponse, Usage


@dataclass
class BlockResult:
    """Result of a single block/LLM call execution."""

    response: LLMCallResponse | None = None
    llm_call_id: UUID | None = None
    usage: Usage | None = None
    error: str | None = None
    error_detail: LLMCallErrorDetail | None = None
    metadata: dict | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.response is not None
