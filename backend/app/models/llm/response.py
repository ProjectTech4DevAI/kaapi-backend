"""
LLM response models.

This module contains structured response models for LLM API calls.
"""
from datetime import datetime
from uuid import UUID
from typing import Literal, Annotated

from sqlmodel import SQLModel, Field

from app.models.llm.request import AudioContent, TextContent


class Usage(SQLModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None


class TextOutput(SQLModel):
    type: Literal["text"] = "text"
    content: TextContent


class AudioOutput(SQLModel):
    type: Literal["audio"] = "audio"
    content: AudioContent


# Type alias for LLM output (discriminated union)
LLMOutput = Annotated[TextOutput | AudioOutput, Field(discriminator="type")]


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
    output: LLMOutput | None = Field(
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


class LLMChainResponse(SQLModel):
    """Response schema for an LLM chain execution."""

    response: LLMResponse = Field(
        ..., description="LLM response from the final step of the chain execution."
    )
    usage: Usage = Field(
        ...,
        description="Aggregate token usage and cost for the entire chain execution.",
    )
    provider_raw_response: dict[str, object] | None = Field(
        default=None,
        description="Raw provider response from the last block (if requested)",
    )


class IntermediateChainResponse(SQLModel):
    """
    Intermediate callback response from the intermediate blocks
    from the llm chain execution. (if configured)

    Flattened structure matching LLMCallResponse keys for consistency
    """

    type: Literal["intermediate"] = "intermediate"
    block_index: int = Field(..., description="Current block position")
    total_blocks: int = Field(..., description="Total number of blocks in the chain")
    response: LLMResponse = Field(
        ..., description="LLM Response from the current block"
    )
    usage: Usage = Field(
        ..., description="Token usage and cost information from the current block"
    )
    provider_raw_response: dict[str, object] | None = Field(
        default=None,
        description="Unmodified raw response from the LLM provider from the current block",
    )


# Job response models
class LLMJobImmediatePublic(SQLModel):
    """Immediate response after creating an LLM job."""

    job_id: UUID
    # status: str
    message: str
    job_inserted_at: datetime
    job_updated_at: datetime
    deleted_at: datetime


class LLMJobPublic(SQLModel):
    """Full job response with nested LLM response when complete."""

    job_id: UUID
    status: str
    # llm_response: LLMCallResponse | None = None
    error_message: str | None = None
