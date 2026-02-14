from typing import Annotated, Any, Literal, Union

from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel
from pydantic import Discriminator, model_validator, HttpUrl
from datetime import datetime
from app.core.util import now

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel, Index, text


class TextLLMParams(SQLModel):
    model: str
    instructions: str | None = Field(
        default=None,
    )
    knowledge_base_ids: list[str] | None = Field(
        default=None,
        description="List of vector store IDs to use for knowledge retrieval",
    )
    reasoning: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="Reasoning configuration or instructions",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
    )
    max_num_results: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of candidate results to return",
    )


class STTLLMParams(SQLModel):
    model: str
    instructions: str
    input_language: str | None = None
    output_language: str | None = None
    response_format: Literal["text"] | None = Field(
        None,
        description="Currently supports text type",
    )
    temperature: float | None = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
    )


class TTSLLMParams(SQLModel):
    model: str
    voice: str
    language: str
    response_format: Literal["mp3", "wav", "ogg"] | None = "wav"


KaapiLLMParams = Union[TextLLMParams, STTLLMParams, TTSLLMParams]


# Input type models for discriminated union
class TextContent(SQLModel):
    format: Literal["text"] = "text"
    value: str = Field(..., description="Text content")


class AudioContent(SQLModel):
    format: Literal["base64"] = "base64"
    value: str = Field(..., min_length=1, description="Base64 encoded audio")
    # keeping the mime_type liberal here, since does not affect transcription type
    mime_type: str | None = Field(
        None,
        description="MIME type of the audio (e.g., audio/wav, audio/mp3, audio/ogg)",
    )


class TextInput(SQLModel):
    type: Literal["text"] = "text"
    content: TextContent


class AudioInput(SQLModel):
    type: Literal["audio"] = "audio"
    content: AudioContent


# Discriminated union for query input types
QueryInput = Annotated[
    Union[TextInput, AudioInput],
    Field(discriminator="type"),
]


class ConversationConfig(SQLModel):
    id: str | None = Field(
        default=None,
        description=(
            "Identifier for an existing conversation. "
            "Used to retrieve the previous message context and continue the chat. "
            "If not provided and `auto_create` is True, a new conversation will be created."
        ),
    )
    auto_create: bool = Field(
        default=False,
        description=(
            "Only if True and no `id` is provided, a new conversation will be created automatically."
        ),
    )

    @model_validator(mode="after")
    def validate_conversation_logic(self):
        if self.id and self.auto_create:
            raise ValueError(
                "Cannot specify both 'id' and 'auto_create=True'. "
                "Use 'id' to continue an existing conversation, or set 'auto_create=True' to create a new one."
            )
        return self


# Query Parameters (dynamic per request)
class QueryParams(SQLModel):
    """Query-specific parameters for each LLM call."""

    input: str | QueryInput = Field(
        ...,
        description=(
            "User input - either a plain string (text) or a structured input object. "
        ),
    )
    conversation: ConversationConfig | None = Field(
        default=None,
        description="Conversation control configuration for context handling.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Normalize plain string input to TextInput for consistency."""
        if isinstance(data, dict) and "input" in data:
            input_val = data["input"]
            if isinstance(input_val, str):
                data["input"] = {
                    "type": "text",
                    "content": {"format": "text", "value": input_val},
                }
        return data


class NativeCompletionConfig(SQLModel):
    """
    Native provider configuration (pass-through).
    All parameters are forwarded as-is to the provider's API without transformation.
    Supports any LLM provider's native API format.
    """

    provider: Literal["openai-native", "google-native"] = Field(
        ...,
        description="Native provider type (e.g., openai-native)",
    )
    params: dict[str, Any] = Field(
        ...,
        description="Provider-specific parameters (schema varies by provider), should exactly match the provider's endpoint params structure",
    )
    type: Literal["text", "stt", "tts"] = Field(
        ..., description="Completion config type. Params schema varies by type"
    )


class KaapiCompletionConfig(SQLModel):
    """
    Kaapi abstraction for LLM completion providers.
    Uses standardized Kaapi parameters that are mapped to provider-specific APIs internally.
    Supports multiple providers: OpenAI, Claude, Gemini, etc.
    """

    provider: Literal["openai", "google"] = Field(
        ..., description="LLM provider (openai)"
    )

    type: Literal["text", "stt", "tts"] = Field(
        ..., description="Completion config type. Params schema varies by type"
    )
    params: dict[str, Any] = Field(
        ...,
        description="Kaapi-standardized parameters mapped to provider-specific API",
    )

    # validate all these 3 config types
    @model_validator(mode="after")
    def validate_params(self):
        param_models = {
            "text": TextLLMParams,
            "stt": STTLLMParams,
            "tts": TTSLLMParams,
        }
        model_class = param_models[self.type]
        validated = model_class.model_validate(self.params)
        self.params = validated.model_dump(exclude_none=True)
        return self


# Discriminated union for completion configs based on provider field
CompletionConfig = Annotated[
    Union[NativeCompletionConfig, KaapiCompletionConfig],
    Field(discriminator="provider"),
]


class ConfigBlob(SQLModel):
    """Raw JSON blob of config."""

    completion: CompletionConfig = Field(..., description="Completion configuration")
    # Future additions:
    # classifier: ClassifierConfig | None = None
    # pre_filter: PreFilterConfig | None = None


class LLMCallConfig(SQLModel):
    """
    Complete configuration for LLM call including all processing stages.
    Either references a stored config (id + version) or provides an ad-hoc config blob.
    Depending on which is provided, only one of the two options should be used.
    """

    id: UUID | None = Field(
        default=None,
        description=(
            "Identifier for an existing LLM call configuration. [require version if provided]"
        ),
    )
    version: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Version of the stored config to use. [require if id is provided]"
        ),
    )

    blob: ConfigBlob | None = Field(
        default=None,
        description=(
            "Raw JSON blob of the full configuration. Used for ad-hoc configurations without storing."
            "Either this or (id + version) must be provided."
        ),
    )

    @model_validator(mode="after")
    def validate_config_logic(self):
        has_stored = self.id is not None or self.version is not None
        has_blob = self.blob is not None

        if has_stored and has_blob:
            raise ValueError(
                "Provide either 'id' with 'version' for stored config OR 'blob' for ad-hoc config, not both."
            )

        if has_stored:
            if not self.id or not self.version:
                raise ValueError(
                    "'id' and 'version' must both be provided together for stored config."
                )
            return self

        if not has_blob:
            raise ValueError(
                "Must provide either a stored config (id + version) or an ad-hoc config (blob)."
            )

        return self

    @property
    def is_stored_config(self) -> bool:
        """Check if the config refers to a stored config or not."""
        return self.id is not None and self.version is not None


class LLMCallRequest(SQLModel):
    """
    API request for an LLM completion.

    The `config` field accepts either:
    - **Stored config (id + version)** — recommended for all production use.
    - **Inline config blob** — for testing or validating new configs.

    Prefer stored configs in production; use blobs only for development/testing/validations.
    """

    query: QueryParams = Field(..., description="Query-specific parameters")
    config: LLMCallConfig = Field(
        ...,
        description=(
            "Complete LLM call configuration, provided either by reference (id + version) "
            "or as config blob. Use the blob only for testing/validation; "
            "in production, always use the id + version."
        ),
    )
    input_guardrails: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional guardrails configuration to apply input validation. "
            "If not provided, no guardrails will be applied."
        ),
    )
    output_guardrails: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional guardrails configuration to apply output validation. "
            "If not provided, no guardrails will be applied."
        ),
    )
    callback_url: HttpUrl | None = Field(
        default=None, description="Webhook URL for async response delivery"
    )
    include_provider_raw_response: bool = Field(
        default=False,
        description="Whether to include the raw LLM provider response in the output",
    )
    request_metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Client-provided metadata passed through unchanged in the response. "
            "Use this to correlate responses with requests or track request state. "
            "The exact dictionary provided here will be returned in the response metadata field."
        ),
    )


class LlmCall(SQLModel, table=True):
    """
    Database model for tracking LLM API call requests and responses.

    Stores both request inputs and response outputs for traceability,
    supporting multimodal inputs (text, audio, image) and various completion types.
    """

    __tablename__ = "llm_call"
    __table_args__ = (
        Index(
            "idx_llm_call_job_id",
            "job_id",
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index(
            "idx_llm_call_conversation_id",
            "conversation_id",
            postgresql_where=text("conversation_id IS NOT NULL AND deleted_at IS NULL"),
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the LLM call record"},
    )

    job_id: UUID = Field(
        foreign_key="job.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={
            "comment": "Reference to the parent job (status tracked in job table)"
        },
    )

    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={
            "comment": "Reference to the project this LLM call belongs to"
        },
    )

    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={
            "comment": "Reference to the organization this LLM call belongs to"
        },
    )

    # Request fields
    input: str = Field(
        ...,
        sa_column_kwargs={
            "comment": "User input - text string, binary data, or file path for multimodal"
        },
    )

    input_type: Literal["text", "audio", "image"] = Field(
        ...,
        sa_column=sa.Column(
            sa.String,
            nullable=False,
            comment="Input type: text, audio, image",
        ),
    )

    output_type: Literal["text", "audio", "image"] | None = Field(
        default=None,
        sa_column=sa.Column(
            sa.String,
            nullable=True,
            comment="Expected output type: text, audio, image",
        ),
    )

    # Provider and model info
    provider: str = Field(
        ...,
        sa_column=sa.Column(
            sa.String,
            nullable=False,
            comment="AI provider as sent by user (e.g openai, -native, google)",
        ),
    )

    model: str = Field(
        ...,
        sa_column_kwargs={
            "comment": "Specific model used e.g. 'gpt-4o', 'gemini-2.5-pro'"
        },
    )

    # Response fields
    provider_response_id: str | None = Field(
        default=None,
        sa_column_kwargs={
            "comment": "Original response ID from the provider (e.g., OpenAI's response ID)"
        },
    )

    content: dict[str, Any] | None = Field(
        default=None,
        sa_column=sa.Column(
            JSONB,
            nullable=True,
            comment="Response content: {text: '...'}, {audio_bytes: '...'}, or {image: '...'}",
        ),
    )

    usage: dict[str, Any] | None = Field(
        default=None,
        sa_column=sa.Column(
            JSONB,
            nullable=True,
            comment="Token usage: {input_tokens, output_tokens, reasoning_tokens}",
        ),
    )

    # Conversation tracking
    conversation_id: str | None = Field(
        default=None,
        sa_column_kwargs={
            "comment": "Identifier linking this response to its conversation thread"
        },
    )

    auto_create: bool | None = Field(
        default=None,
        sa_column_kwargs={
            "comment": "Whether to auto-create conversation if conversation_id doesn't exist (OpenAI specific)"
        },
    )

    # Configuration - stores either {config_id, config_version} or {config_blob}
    config: dict[str, Any] | None = Field(
        default=None,
        sa_column=sa.Column(
            JSONB,
            nullable=True,
            comment="Configuration: {config_id, config_version} for stored config OR {config_blob} for ad-hoc config",
        ),
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the LLM call was created"},
    )

    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the LLM call was last updated"},
    )

    deleted_at: datetime | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Timestamp when the record was soft-deleted"},
    )
