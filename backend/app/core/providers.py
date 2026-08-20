import logging
from typing import Annotated, Any
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Enumeration of supported credential providers."""

    OPENAI = "openai"
    LANGFUSE = "langfuse"
    GOOGLE_AISTUDIO = "google-aistudio"
    SARVAMAI = "sarvamai"
    ELEVENLABS = "elevenlabs"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    WEBHOOK_SECRET = "webhook_secret"
    PROXY = "proxy"


CredentialPayload = dict[str, JsonValue]


class ProviderCredentialsBase(BaseModel):
    """Base for provider credential payloads.

    Extra keys are allowed and preserved. Callers already store keys beyond the
    declared auth fields, and the payload is written back verbatim, so
    forbidding them would reject requests that work today. Tightening this to
    ``extra="forbid"`` is a deliberate breaking change and needs its own
    migration for existing rows.
    """

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


# Provider configurations
PROVIDER_CONFIGS: dict[Provider, ProviderConfig] = {
    Provider.OPENAI: ProviderConfig(
        model=OpenAICredentials, sensitive_fields=["api_key"]
    ),
    Provider.LANGFUSE: ProviderConfig(
        model=LangfuseCredentials,
        sensitive_fields=["secret_key"],
    ),
    Provider.GOOGLE_AISTUDIO: ProviderConfig(
        model=GoogleAIStudioCredentials, sensitive_fields=["api_key"]
    ),
    Provider.SARVAMAI: ProviderConfig(
        model=SarvamAICredentials, sensitive_fields=["api_key"]
    ),
    Provider.ELEVENLABS: ProviderConfig(
        model=ElevenLabsCredentials, sensitive_fields=["api_key"]
    ),
    Provider.ANTHROPIC: ProviderConfig(
        model=AnthropicCredentials, sensitive_fields=["api_key"]
    ),
    Provider.GOOGLE: ProviderConfig(
        model=GoogleCredentials,
        sensitive_fields=["api_key"],
    ),
    Provider.WEBHOOK_SECRET: ProviderConfig(
        model=WebhookSecretCredentials, sensitive_fields=["webhook_secret"]
    ),
    Provider.PROXY: ProviderConfig(
        model=ProxyCredentials, sensitive_fields=["api_key"]
    ),
}


def validate_provider(provider: str) -> Provider:
    """Validate that the provider name is supported and return the Provider enum.

    Args:
        provider: The provider name to validate

    Returns:
        Provider: The validated provider enum

    Raises:
        ValueError: If the provider is not supported
    """
    try:
        return Provider(provider.lower())
    except (AttributeError, ValueError):  # non-string keys reach here too
        supported = ", ".join(p.value for p in Provider)
        logger.warning(
            f"[validate_provider] Unsupported provider | provider: {provider}, supported_providers: {supported}"
        )
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers are: {supported}"
        )


def parse_provider_credentials(
    provider: str, credentials: Any
) -> ProviderCredentialsBase:
    """Validate a raw credential payload and return it as the provider's model.

    Raises:
        ValueError: If the provider is unsupported, the payload is not an
            object, a required field is missing, a field has the wrong type, or
            the payload carries keys the provider does not declare
    """
    provider_enum = validate_provider(provider)

    if not isinstance(credentials, dict):
        raise ValueError(f"Value for provider '{provider}' must be an object/dict.")

    config = PROVIDER_CONFIGS[provider_enum]

    # Checked up front so the message stays field-oriented; Pydantic would
    # otherwise report one "Field required" error per missing key.
    if missing_fields := [
        name for name in config.required_fields if name not in credentials
    ]:
        logger.warning(
            f"[parse_provider_credentials] Missing required fields | provider: {provider}, missing_fields: {', '.join(missing_fields)}"
        )
        raise ValueError(
            f"Missing required fields for {provider}: {', '.join(missing_fields)}"
        )

    try:
        return config.model.model_validate(credentials)
    except ValidationError as e:
        invalid_fields = ", ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        logger.warning(
            f"[parse_provider_credentials] Invalid credential fields | provider: {provider}, errors: {invalid_fields}"
        )
        raise ValueError(f"Invalid credentials for {provider}: {invalid_fields}")


def mask_credential_fields(
    provider: str, credentials: CredentialPayload
) -> CredentialPayload:
    """Mask sensitive fields in a credential dict for the given provider.

    Non-sensitive fields (e.g., langfuse `public_key`, `host`) are returned as-is.
    Unknown providers are returned with no masking.
    """
    from app.utils import mask_string

    if not credentials:
        return credentials

    try:
        provider_enum = Provider(provider.lower())
    except ValueError:
        return credentials

    sensitive_fields = PROVIDER_CONFIGS[provider_enum].sensitive_fields
    masked = dict(credentials)
    for field_name in sensitive_fields:
        if field_name not in masked:
            continue
        value = masked[field_name]
        if isinstance(value, str):
            masked[field_name] = mask_string(value)
        else:
            # Non-string secrets (e.g. ``google`` Vertex `sa_key` is a dict)
            # are masked wholesale — the raw value is only decrypted at
            # provider runtime, never returned via the API.
            masked[field_name] = "********"
    return masked
