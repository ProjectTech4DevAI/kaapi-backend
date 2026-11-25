"""Request models for the speech-to-speech endpoint (`/llm/chain/sts`)."""

from typing import Any, Self
from uuid import UUID

from pydantic import HttpUrl, model_validator
from sqlmodel import Field, SQLModel

from app.models.llm.constants import Provider, RAGProvider, STTProvider, TTSProvider
from app.models.llm.request import (
    AudioInput,
    ConversationConfig,
    QueryParams,
    STTLLMParams,
    TextLLMParams,
    TTSLLMParams,
)


class _StepConfigBase(SQLModel):
    """Either a stored config (id + version) or inline provider/params, not both."""

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

    @model_validator(mode="after")
    def _validate_xor(self) -> Self:
        has_ref = self.id is not None or self.version is not None
        has_inline = (
            getattr(self, "provider", None) is not None
            or getattr(self, "params", None) is not None
        )

        if has_ref and has_inline:
            raise ValueError(
                "Provide either (id + version) OR inline 'provider'/'params', not both."
            )
        if has_ref and (self.id is None or self.version is None):
            raise ValueError("Both 'id' and 'version' must be set together.")
        return self

    @property
    def is_stored_ref(self) -> bool:
        return self.id is not None and self.version is not None


class STTStepConfig(_StepConfigBase):
    provider: STTProvider | None = Field(
        default=None,
        description="STT provider. Omit to auto-select the configured Google credential.",
    )
    params: STTLLMParams | None = Field(
        default=None,
        description="Inline STT parameters, used as written. Omit to use the params defaults.",
    )


class LLMStepConfig(_StepConfigBase):
    provider: RAGProvider | None = Field(
        default=None, description="Text LLM provider. Defaults to 'openai'."
    )
    params: TextLLMParams | None = Field(
        default=None,
        description="Inline text LLM parameters, used as written. Omit to use the params defaults.",
    )
    conversation: ConversationConfig | None = Field(
        default=None,
        description=(
            "Continue a conversation (`id`) or start one (`auto_create`), as in "
            "/llm/call `query.conversation`. OpenAI only."
        ),
    )

    @model_validator(mode="after")
    def validate_provider_supports_conversation(self) -> Self:
        # Stored steps are checked in the route, once their provider is known.
        if self.conversation is not None and self.provider not in (
            None,
            Provider.OPENAI,
        ):
            raise ValueError(
                f"conversation is not supported with provider '{self.provider}'"
            )
        return self

    @model_validator(mode="after")
    def validate_provider_supports_kb(self) -> Self:
        # Vertex has no Gemini file-search stores, so KB ids would be silently dropped.
        if (
            self.provider == Provider.GOOGLE_GCP
            and self.params is not None
            and self.params.knowledge_base_ids
        ):
            raise ValueError(
                "knowledge_base_ids is not supported with provider 'google-gcp'"
            )
        return self


class TTSStepConfig(_StepConfigBase):
    provider: TTSProvider | None = Field(
        default=None,
        description="TTS provider. Omit to auto-select the configured Google credential.",
    )
    params: TTSLLMParams | None = Field(
        default=None,
        description="Inline TTS parameters, used as written. Omit to use the params defaults.",
    )


class SpeechToSpeechConfig(SQLModel):
    stt: STTStepConfig | None = Field(
        default=None,
        description="STT step: stored (id + version) or inline (provider + params). Omit for endpoint defaults.",
    )
    llm: LLMStepConfig | None = Field(
        default=None,
        description="Text LLM step: stored (id + version) or inline (provider + params). Omit for endpoint defaults.",
    )
    tts: TTSStepConfig | None = Field(
        default=None,
        description="TTS step: stored (id + version) or inline (provider + params). Omit for endpoint defaults.",
    )


class SpeechToSpeechRequest(SQLModel):
    """
    API request for speech-to-speech (STS): a 3-block STT -> LLM -> TTS chain.

    Each step in `config` is either a stored config (id + version) or an inline
    provider + params; omitted steps fall back to endpoint defaults.
    """

    query: QueryParams = Field(
        ..., description="Query parameters; `input` must be a single audio input"
    )
    config: SpeechToSpeechConfig | None = Field(
        default=None, description="Per-step configuration for stt, llm and tts"
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

    @model_validator(mode="after")
    def validate_audio_input(self) -> Self:
        if not isinstance(self.query.input, AudioInput):
            raise ValueError("query.input must be a single audio input")
        # The STT block would receive it and drop it; the LLM step owns conversations.
        if self.query.conversation is not None:
            raise ValueError("Set conversation on config.llm, not on query")
        return self
