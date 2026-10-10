"""LLM prompt config payloads proxied to the kaapi-guardrails service.

Mirrors `kaapi-guardrails/backend/app/schemas/llm_prompt_config.py` at the
level of field names and types only. Length rules and the `{query}`/`{answer}`
placeholder rule are left to the upstream service, which owns validation.

Timestamps keep the upstream ``created_at``/``updated_at`` spelling because
these objects are echoed verbatim from the other service.
"""

from datetime import datetime
from uuid import UUID

from sqlmodel import Field, SQLModel

from app.models.guardrails.enums import LLMValidatorNameEnum


class LLMPromptConfigCreate(SQLModel):
    """Request body for ``POST /api/v1/guardrails/llm_prompt_configs``."""

    validator_name: LLMValidatorNameEnum = Field(
        ...,
        description=(
            "Which LLM-backed validator this prompt drives. Cannot be changed "
            "after creation."
        ),
    )
    name: str = Field(..., description="Human-readable name.")
    description: str = Field(..., description="What this prompt evaluates.")
    prompt_schema_version: int = Field(
        default=1,
        description="Version of the prompt contract, for callers that iterate on wording.",
    )
    llm_prompt: str = Field(
        ...,
        description=(
            "The prompt text. For `answer_relevance_custom_llm` this must "
            "contain both the `{query}` and `{answer}` placeholders; the "
            "guardrails service rejects it with a 422 otherwise."
        ),
    )


class LLMPromptConfigUpdate(SQLModel):
    """Request body for ``PATCH /guardrails/llm_prompt_configs/{prompt_config_id}``.

    ``validator_name`` is absent by design — it is immutable after creation.
    """

    name: str | None = None
    description: str | None = None
    prompt_schema_version: int | None = None
    llm_prompt: str | None = None
    is_active: bool | None = Field(
        default=None,
        description="Only settable on update. New configs are created active.",
    )


class LLMPromptConfigPublic(SQLModel):
    """An LLM prompt config as returned by the guardrails service."""

    id: UUID
    validator_name: LLMValidatorNameEnum
    name: str
    description: str
    prompt_schema_version: int
    llm_prompt: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
