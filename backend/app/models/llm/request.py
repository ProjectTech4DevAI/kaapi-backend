from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Self, Union
from uuid import UUID, uuid4

import sqlalchemy as sa
from pydantic import HttpUrl, model_validator
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Index, SQLModel, text

from app.core.util import now
from app.models.llm.constants import (
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    CompletionType,
    KaapiProvider,
    NativeProvider,
    Provider,
    RAGProvider,
    STTProvider,
    TTSProvider,
)


class TextLLMParams(SQLModel):
    model: str | None = Field(
        default=None,
        description=(
            "Provider model to use. If omitted, the Kaapi mapper falls back to "
            "DEFAULT_TEXT_MODELS for the selected provider."
        ),
    )
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
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = Field(
        default=None,
        description="Model-specific reasoning effort setting for reasoning-capable models",
    )
    summary: Literal["auto", "detailed", "concise"] | None = Field(
        default=None,
        description=(
            "Model-specific reasoning summary preference. " "Use null/None to disable."
        ),
    )
    temperature: float | None = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate in the response",
    )
    max_num_results: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of candidate results to return",
    )


class STTLLMParams(SQLModel):
    model_config = {"extra": "forbid"}

    model: str = DEFAULT_STT_MODEL
    instructions: str | None = None
    input_language: str | None = "auto"
    output_language: str | None = None
    response_format: Literal["text"] | None = Field(
        None,
        description="Currently supports text type",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.01,  # sarvam minimum 0.01
        le=2.0,
        description="Temperature parameter (not supported by all STT providers)",
    )


class TTSLLMParams(SQLModel):
    model_config = {"extra": "forbid"}

    model: str = DEFAULT_TTS_MODEL
    voice: str = DEFAULT_TTS_VOICE
    language: str | None = None
    response_format: Literal["mp3", "wav", "ogg"] | None = "wav"
    instructions: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _reject_nonempty_instructions(self) -> Self:
        if self.instructions:
            raise ValueError("instructions is not supported for TTS completions")
        return self


class ProxyLLMParams(SQLModel):
    model_config = {"extra": "forbid"}

    client_llm_url: HttpUrl = Field(
        ...,
        description=(
            "HTTPS URL of the client's own LLM endpoint. Kaapi forwards the "
            "(guardrail-sanitised) input here and applies output guardrails to the response."
        ),
    )

    @model_validator(mode="after")
    def _require_https(self) -> Self:
        if self.client_llm_url.scheme != "https":
            raise ValueError(
                f"client_llm_url must be HTTPS, got scheme: {self.client_llm_url.scheme}"
            )
        return self


KaapiLLMParams = Union[TextLLMParams, STTLLMParams, TTSLLMParams, ProxyLLMParams]


# Input type models for discriminated union
class TextContent(SQLModel):
    format: Literal["text"] = "text"
    value: str = Field(..., description="Text content")
    language_code: str | None = Field(
        None, description="Optional detected language code in STT 'auto' mode"
    )


class AudioContent(SQLModel):
    format: Literal["base64", "url"] = "base64"
    value: str = Field(
        ..., description="Base64 encoded audio or public URL to download from"
    )
    # keeping the mime_type liberal here, since does not affect base64 encoding
    mime_type: str | None = Field(
        None,
        description="MIME type of the audio (e.g., audio/wav, audio/mp3, audio/ogg)",
    )
    uri: str | None = Field(
        None,
        description="Presigned URL to the audio file in object storage (when available)",
    )


class ImageContent(SQLModel):
    format: Literal["base64", "url"] = "base64"
    value: str = Field(
        ..., description="Base64 encoded image or Public URL to the image"
    )
    # keeping the mime_type
    mime_type: str | None = Field(
        None,
        description="MIME type of the image (e.g., image/png, image/jpeg)",
    )


class PDFContent(SQLModel):
    format: Literal["base64", "url"] = "base64"
    value: str = Field(..., description="Base64 encoded PDF or Public URL to the PDF")
    # keeping the mime_type
    mime_type: str | None = Field(
        None,
        description="MIME type of the PDF (e.g., application/pdf)",
    )


class TextInput(SQLModel):
    type: Literal["text"] = "text"
    content: TextContent


class AudioInput(SQLModel):
    type: Literal["audio"] = "audio"
    content: AudioContent


class ImageInput(SQLModel):
    type: Literal["image"] = "image"
    content: ImageContent | list[ImageContent]


class PDFInput(SQLModel):
    type: Literal["pdf"] = "pdf"
    content: PDFContent | list[PDFContent]


# Discriminated union for query input types
QueryInput = Annotated[
    Union[TextInput, AudioInput, ImageInput, PDFInput],
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

    input: str | QueryInput | list[QueryInput] = Field(
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

    provider: NativeProvider = Field(
        ...,
        description="Native provider type (e.g., openai-native)",
    )
    params: dict[str, Any] = Field(
        ...,
        description="Provider-specific parameters (schema varies by provider), should exactly match the provider's endpoint params structure",
    )
    type: CompletionType = Field(
        ..., description="Completion config type. Params schema varies by type"
    )


class KaapiCompletionConfig(SQLModel):
    """
    Kaapi abstraction for LLM completion providers.
    Uses standardized Kaapi parameters that are mapped to provider-specific APIs internally.
    Supports multiple providers: OpenAI, Claude, Gemini, etc.
    """

    provider: KaapiProvider | None = Field(
        None,
        description=(
            "LLM provider (openai, google, sarvamai, elevenlabs, anthropic, "
            "google-aistudio). 'google' routes via Google Vertex AI; "
            "'google-aistudio' uses Google AI Studio."
        ),
    )

    type: CompletionType = Field(
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

        if (
            self.type in (CompletionType.STT, CompletionType.TTS)
            and self.provider is None
        ):
            self.provider = Provider.GOOGLE

        user_provided_temperature = "temperature" in self.params
        validated = model_class.model_validate(self.params)

        self.params = validated.model_dump(exclude_none=True)
        if not user_provided_temperature:
            self.params.pop("temperature", None)
        return self


class ProxyCompletionConfig(SQLModel):
    """
    Proxy completion: Kaapi forwards the (guardrail-sanitised) input to the
    client's own LLM endpoint and applies output guardrails to the response.
    No upstream provider is dispatched — `provider` is fixed to "proxy" so
    the discriminated union can route cleanly.
    """

    provider: Literal["proxy"] = Field(
        "proxy",
        description=(
            "Discriminator value for the proxy variant. Auto-injected when "
            "type=proxy; clients may omit it."
        ),
    )
    type: Literal["proxy"] = Field(..., description="Must be 'proxy'.")
    params: dict[str, Any] = Field(
        ...,
        description="Proxy params (client_llm_url, ...)",
    )

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        validated = ProxyLLMParams.model_validate(self.params)
        # mode="json" coerces HttpUrl → plain str so downstream consumers
        # (httpx.post, urlparse) get the type they expect from params dict.
        self.params = validated.model_dump(mode="json", exclude_none=True)
        return self


# Discriminated union for completion configs based on provider field
CompletionConfig = Annotated[
    Union[NativeCompletionConfig, KaapiCompletionConfig, ProxyCompletionConfig],
    Field(discriminator="provider"),
]


class Validator(SQLModel):
    validator_config_id: UUID


class PromptTemplate(SQLModel):
    template: str = Field(..., description="Template string with {{input}} placeholder")


class ConfigBlob(SQLModel):
    """Raw JSON blob of config."""

    completion: CompletionConfig = Field(..., description="Completion configuration")

    @model_validator(mode="before")
    @classmethod
    def _default_proxy_provider(cls, data: Any) -> Any:
        """For `type=proxy`, provider is meaningless to the caller.
        Inject provider="proxy" so the CompletionConfig discriminator routes
        to ProxyCompletionConfig without forcing the client to set it."""
        if not isinstance(data, dict):
            return data
        completion = data.get("completion")
        if (
            isinstance(completion, dict)
            and completion.get("type") == Provider.PROXY.value
        ):
            existing = completion.get("provider")
            if existing in (None, Provider.PROXY.value):
                completion["provider"] = Provider.PROXY.value
        return data

    # used for llm-chain to provide prompt interpolation
    prompt_template: PromptTemplate | None = Field(
        default=None,
        description="Prompt template with {{input}} placeholder to wrap around the user input",
    )

    input_guardrails: list[Validator] | None = Field(
        default=None,
        description="Guardrails applied to validate/sanitize the input before the LLM call",
    )

    output_guardrails: list[Validator] | None = Field(
        default=None,
        description="Guardrails applied to validate/sanitize the output after the LLM call",
    )
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
        Index(
            "idx_llm_call_chain_id",
            "chain_id",
            postgresql_where=text("chain_id IS NOT NULL"),
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

    chain_id: UUID | None = Field(
        default=None,
        foreign_key="llm_chain.id",
        nullable=True,
        ondelete="SET NULL",
        sa_column_kwargs={
            "comment": "Reference to the parent chain (NULL for standalone llm_call requests)"
        },
    )

    # Request fields
    input: str = Field(
        ...,
        sa_column_kwargs={
            "comment": "User input - text string, binary data, or file path for multimodal"
        },
    )

    # NOTE: image, pdf, multimodal are internal labels stored in the table not user facing.
    input_type: Literal["text", "audio", "image", "pdf", "multimodal"] = Field(
        ...,
        sa_column=sa.Column(
            sa.String,
            nullable=False,
            comment="Input type: text, audio, image, pdf, multimodal",
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
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the LLM call was created"},
    )

    updated_at: datetime = Field(
        default_factory=now,
        sa_column=sa.Column(
            sa.DateTime,
            default=now,
            nullable=False,
            onupdate=now,
            comment="Timestamp when the LLM call was last updated",
        ),
    )

    deleted_at: datetime | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Timestamp when the record was soft-deleted"},
    )


class ChainBlock(SQLModel):
    """A single block in an LLM chain execution."""

    config: LLMCallConfig = Field(
        ..., description="LLM call configuration (stored id+version OR ad-hoc blob)"
    )

    include_provider_raw_response: bool = Field(
        default=False,
        description="Whether to include the raw LLM provider response in the output for this block",
    )

    intermediate_callback: bool = Field(
        default=False,
        description="Whether to send intermediate callback after this block completes",
    )


class LLMChainRequest(SQLModel):
    """
    API request for an LLM chain execution.

    Orchestrates multiple LLM calls sequentially where each block's output
    becomes the next block's input.
    """

    query: QueryParams = Field(
        ..., description="Initial query input for the first block in the chain"
    )

    blocks: list[ChainBlock] = Field(
        ..., min_length=1, description="Ordered list of blocks to execute sequentially"
    )

    callback_url: HttpUrl | None = Field(
        default=None, description="Webhook URL for async response delivery"
    )

    request_metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Client-provided metadata passed through unchanged in the response. "
            "Use this to correlate responses with requests or track request state. "
            "The exact dictionary provided here will be returned in the response metadata field."
        ),
    )


class ChainStatus(str, Enum):
    """Status of an LLM chain execution."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class LlmChain(SQLModel, table=True):
    """
    Database model for tracking LLM chain execution

    it manages and orchestrates sequential llm_call executions.
    """

    __tablename__ = "llm_chain"
    __table_args__ = (
        Index(
            "idx_llm_chain_job_id",
            "job_id",
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the LLM chain record"},
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

    status: ChainStatus = Field(
        default=ChainStatus.PENDING,
        sa_column_kwargs={
            "comment": "Chain execution status (pending, running, failed, completed)"
        },
    )

    error: str | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Error message if the chain execution failed"},
    )

    block_sequences: list[str] | None = Field(
        default_factory=list,
        sa_column=sa.Column(
            JSONB,
            nullable=True,
            comment="Ordered list of llm_call UUIDs as blocks complete",
        ),
    )

    total_blocks: int = Field(
        ..., sa_column_kwargs={"comment": "Total number of blocks to execute"}
    )

    number_of_blocks_processed: int = Field(
        default=0,
        sa_column_kwargs={
            "comment": "Number of blocks processed so far (used for tracking progress)"
        },
    )

    # Request fields
    input: str = Field(
        ...,
        sa_column_kwargs={
            "comment": "First block user's input - text string, binary data, or file path for multimodal"
        },
    )

    output: dict[str, Any] | None = Field(
        default=None,
        sa_column=sa.Column(
            JSONB,
            nullable=True,
            comment="Last block's final output (set on chain completion)",
        ),
    )

    configs: list[dict[str, Any]] | None = Field(
        default=None,
        sa_column=sa.Column(
            JSONB,
            nullable=True,
            comment="Ordered list of block configs as submitted in the request",
        ),
    )

    total_usage: dict[str, Any] | None = Field(
        default=None,
        sa_column=sa.Column(
            JSONB,
            nullable=True,
            comment="Aggregated token usage: {input_tokens, output_tokens, total_tokens}",
        ),
    )

    metadata_: dict[str, Any] | None = Field(
        default=None,
        sa_column=sa.Column(
            "metadata",
            JSONB,
            nullable=True,
            comment="Future-proof extensibility catch-all",
        ),
    )

    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the chain record was created"},
    )

    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the chain record was last updated"
        },
    )


class _BlockSpecBase(SQLModel):
    """Common xor logic for per-block specs: provide *either* (config_id +
    config_version) for a stored config, *or* inline `params`. Not both.
    """

    config_id: UUID | None = Field(
        default=None,
        description="ID of a stored LLM config to use for this block.",
    )
    config_version: int | None = Field(
        default=None,
        ge=1,
        description="Version of the stored config (required when config_id is set).",
    )

    @model_validator(mode="after")
    def _validate_xor(self):
        has_ref = self.config_id is not None or self.config_version is not None
        has_params = getattr(self, "params", None) is not None

        if has_ref and has_params:
            raise ValueError(
                "Provide either (config_id + config_version) OR inline 'params', not both."
            )
        if has_ref and (self.config_id is None or self.config_version is None):
            raise ValueError(
                "Both 'config_id' and 'config_version' must be set together."
            )
        return self

    @property
    def is_stored_ref(self) -> bool:
        return self.config_id is not None and self.config_version is not None


class STTBlockSpec(_BlockSpecBase):
    params: STTLLMParams | None = Field(
        default=None,
        description="Inline STT parameters. Omit to use endpoint defaults.",
    )


class RAGBlockSpec(_BlockSpecBase):
    params: TextLLMParams | None = Field(
        default=None,
        description="Inline RAG (text) parameters. Omit to use endpoint defaults.",
    )


class TTSBlockSpec(_BlockSpecBase):
    params: TTSLLMParams | None = Field(
        default=None,
        description="Inline TTS parameters. Omit to use endpoint defaults.",
    )


class SpeechToSpeechRequest(SQLModel):
    """
    API request for speech-to-speech (STS) with RAG.

    Convenience endpoint that orchestrates a 3-block chain:
    STT → RAG → TTS

    Input: Audio
    Output: Audio + Text (via callback)
    """

    query: AudioInput = Field(
        ..., description="Voice note input (WhatsApp compatible format)"
    )
    knowledge_base_ids: list[str] = Field(
        ..., min_length=1, description="Knowledge base IDs for RAG retrieval"
    )

    # Optional language config (BCP-47 codes)
    input_language: str = Field(
        "auto",
        description=(
            "BCP-47 language code for STT input (auto-detect by default). "
            "Supported codes: 'auto', 'en-IN', 'hi-IN', 'bn-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'od-IN', "
            "'pa-IN', 'ta-IN', 'te-IN', 'gu-IN', 'as-IN', 'ur-IN', 'ne-IN', 'kok-IN', 'ks-IN', "
            "'sd-IN', 'sa-IN', 'sat-IN', 'mni-IN', 'brx-IN', 'mai-IN', 'doi-IN'"
        ),
    )
    output_language: str | None = Field(
        None,
        description=(
            "BCP-47 language code for TTS output (defaults to input_language if not specified). "
            "Supported codes: same as input_language (except 'auto')."
        ),
    )

    # Per-block specs. Each spec accepts EITHER (config_id + config_version)
    # to reference a stored config, OR inline `params` to override the
    # endpoint defaults. Omit entirely to use defaults only.
    stt: STTBlockSpec | None = Field(
        None,
        description=(
            "STT block spec. Use 'params' for inline overrides or "
            "'config_id' + 'config_version' to reference a stored config."
        ),
    )
    rag: RAGBlockSpec | None = Field(
        None,
        description=(
            "RAG block spec. Use 'params' for inline overrides or "
            "'config_id' + 'config_version' to reference a stored config."
        ),
    )
    tts: TTSBlockSpec | None = Field(
        None,
        description=(
            "TTS block spec. Use 'params' for inline overrides or "
            "'config_id' + 'config_version' to reference a stored config."
        ),
    )

    # Provider hints. Optional — KaapiCompletionConfig auto-defaults to
    # "google" for stt/tts when omitted.
    stt_provider: STTProvider | None = None
    tts_provider: TTSProvider | None = None
    rag_provider: RAGProvider | None = None

    # Callback and metadata
    callback_url: HttpUrl | None = Field(
        None, description="Webhook URL for async response delivery"
    )
    request_metadata: dict[str, Any] | None = Field(
        None, description="Client-provided metadata"
    )

    @model_validator(mode="after")
    def validate_languages(self):
        """Normalize BCP-47 language codes to standard format (e.g., 'hi-in' -> 'hi-IN')."""
        # Normalize input_language
        if self.input_language and self.input_language != "auto":
            # Normalize BCP-47: lowercase language, uppercase region (e.g., "hi-IN")
            parts = self.input_language.split("-")
            if len(parts) == 2:
                self.input_language = f"{parts[0].lower()}-{parts[1].upper()}"

        # Normalize output_language
        if self.output_language:
            parts = self.output_language.split("-")
            if len(parts) == 2:
                self.output_language = f"{parts[0].lower()}-{parts[1].upper()}"

        return self
