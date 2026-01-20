from typing import Annotated, Any, Literal, Union

from uuid import UUID
from sqlmodel import Field, SQLModel
from pydantic import Discriminator, model_validator, HttpUrl


class KaapiLLMParams(SQLModel):
    """
    Kaapi-abstracted parameters for LLM providers.
    These parameters are mapped internally to provider-specific API parameters.
    Provides a unified contract across all LLM providers (OpenAI, Claude, Gemini, etc.).
    Provider-specific mappings are handled at the mapper level.
    """

    model: str = Field(
        description="Model identifier to use for completion (e.g., 'gpt-4o', 'gpt-5')",
    )
    instructions: str | None = Field(
        default=None,
        description="System instructions to guide the model's behavior",
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
        description="Sampling temperature between 0 and 2",
    )
    max_num_results: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of results to return",
    )


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

    input: str = Field(
        ...,
        min_length=1,
        description="User input question/query/prompt, used to generate a response.",
    )
    conversation: ConversationConfig | None = Field(
        default=None,
        description="Conversation control configuration for context handling.",
    )


class NativeCompletionConfig(SQLModel):
    """
    Native provider configuration (pass-through).
    All parameters are forwarded as-is to the provider's API without transformation.
    Supports any LLM provider's native API format.
    """

    provider: Literal["openai-native"] = Field(
        default="openai-native",
        description="Native provider type (e.g., openai-native)",
    )
    params: dict[str, Any] = Field(
        ...,
        description="Provider-specific parameters (schema varies by provider), should exactly match the provider's endpoint params structure",
    )


class KaapiCompletionConfig(SQLModel):
    """
    Kaapi abstraction for LLM completion providers.
    Uses standardized Kaapi parameters that are mapped to provider-specific APIs internally.
    Supports multiple providers: OpenAI, Claude, Gemini, etc.
    """

    provider: Literal["openai"] = Field(..., description="LLM provider (openai)")
    params: KaapiLLMParams = Field(
        ...,
        description="Kaapi-standardized parameters mapped to provider-specific API",
    )


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
