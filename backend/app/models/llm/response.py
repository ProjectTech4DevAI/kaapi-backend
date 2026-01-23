"""
LLM response models.

This module contains structured response models for LLM API calls.
"""

from sqlmodel import SQLModel, Field


class Usage(SQLModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None


class LLMOutput(SQLModel):
    """Standardized output format for LLM responses."""

    text: str = Field(..., description="Primary text content of the LLM response.")


class LLMResponse(SQLModel):
    """Normalized response format independent of provider."""

    provider_response_id: str = Field(
        ..., description="Unique response ID provided by the LLM provider."
    )
    conversation_id: str | None = Field(
        default=None, description="Conversation or thread ID for context (if any)."
    )
    provider: str = Field(
        ..., description="Name of the LLM provider (e.g., openai, anthropic)."
    )
    model: str = Field(
        ..., description="Model used by the provider (e.g., gpt-4-turbo)."
    )
    output: LLMOutput = Field(
        ...,
        description="Structured output containing text and optional additional data.",
    )


class LLMCallResponse(SQLModel):
    """Top-level response schema for an LLM API call."""

    response: LLMResponse = Field(
        ..., description="Normalized, structured LLM response."
    )
    usage: Usage = Field(..., description="Token usage and cost information.")
    provider_raw_response: dict[str, object] | None = Field(
        default=None,
        description="Unmodified raw response from the LLM provider.",
    )
