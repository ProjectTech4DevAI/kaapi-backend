from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Provider(str, Enum):
    """Enumeration of supported credential providers."""

    OPENAI = "openai"
    LANGFUSE = "langfuse"
    GOOGLE_AISTUDIO = "google-aistudio"
    GOOGLE_GCP = "google-gcp"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    WEBHOOK_SECRET = "webhook_secret"
    PROXY = "proxy"


CredentialPayload = dict[str, JsonValue]


class ProviderCredentialsBase(BaseModel):
    """Base for provider credential payloads."""

    model_config = ConfigDict(extra="allow")


class OpenAICredentials(ProviderCredentialsBase):
    api_key: str = Field(description="OpenAI API key")


class LangfuseCredentials(ProviderCredentialsBase):
    secret_key: str = Field(description="Langfuse secret key")
    public_key: str = Field(description="Langfuse public key")
    host: str = Field(description="Langfuse host URL")


class GoogleAIStudioCredentials(ProviderCredentialsBase):
    api_key: str = Field(description="Google AI Studio API key")


class SarvamAICredentials(ProviderCredentialsBase):
    api_key: str = Field(description="SarvamAI API key")


class ElevenLabsCredentials(ProviderCredentialsBase):
    api_key: str = Field(description="ElevenLabs API key")


class AnthropicCredentials(ProviderCredentialsBase):
    api_key: str = Field(description="Anthropic API key")


class GoogleCredentials(ProviderCredentialsBase):
    api_key: str = Field(description="Google API key")


class GoogleGcpCredentials(ProviderCredentialsBase):
    """Either ``api_key`` or ``sa_key`` authenticates. API keys hit Vertex's
    projectless endpoint; an SA key carries its own ``project_id``."""

    api_key: str | None = Field(default=None, description="Google GCP API key")
    sa_key: dict[str, JsonValue] | None = Field(
        default=None, description="Service account key JSON object"
    )
    project_id: str | None = Field(
        default=None, description="GCP project ID; defaults to sa_key.project_id"
    )
    location: str | None = Field(
        default=None, description="Vertex location; defaults to 'global'"
    )
    gcs_bucket: str | None = Field(
        default=None,
        description="GCS bucket name; needed with sa_key for STT audio and batch",
    )

    @model_validator(mode="after")
    def _require_auth(self) -> "GoogleGcpCredentials":
        if not (self.api_key or self.sa_key):
            raise ValueError("provide api_key or sa_key")
        # A cross-project SA needs project_id set explicitly to bill elsewhere.
        if self.sa_key and not (self.project_id or self.sa_key.get("project_id")):
            raise ValueError("sa_key has no project_id; set project_id")
        return self


class WebhookSecretCredentials(ProviderCredentialsBase):
    webhook_secret: str = Field(
        description="Shared secret used to HMAC-sign outgoing webhooks"
    )


class ProxyCredentials(ProviderCredentialsBase):
    api_key: str = Field(description="Proxy API key")


ProviderCredentials = Annotated[
    OpenAICredentials
    | LangfuseCredentials
    | GoogleAIStudioCredentials
    | SarvamAICredentials
    | ElevenLabsCredentials
    | AnthropicCredentials
    | GoogleCredentials
    | GoogleGcpCredentials
    | WebhookSecretCredentials
    | ProxyCredentials,
    Field(
        description=(
            "Credential payload. The accepted shape depends on the provider it "
            "is keyed by — see the per-provider schemas."
        )
    ),
]


@dataclass
class ProviderConfig:
    """Configuration for a provider: its credential schema and which of those
    fields are secret."""

    model: type[ProviderCredentialsBase]
    sensitive_fields: list[str] = field(default_factory=list)

    @property
    def required_fields(self) -> list[str]:
        """Credential keys the provider cannot be used without, derived from
        the schema so the two can never drift."""
        return [
            name
            for name, model_field in self.model.model_fields.items()
            if model_field.is_required()
        ]
