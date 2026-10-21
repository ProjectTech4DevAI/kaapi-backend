from pydantic import ValidationError

from app.models.provider_credentials import (
    AnthropicCredentials,
    CredentialPayload,
    ElevenLabsCredentials,
    GoogleAIStudioCredentials,
    GoogleCredentials,
    GoogleGcpCredentials,
    LangfuseCredentials,
    OpenAICredentials,
    Provider,
    ProviderConfig,
    ProviderCredentialsBase,
    ProxyCredentials,
    SarvamAICredentials,
    WebhookSecretCredentials,
)

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
    Provider.GOOGLE_GCP: ProviderConfig(
        model=GoogleGcpCredentials,
        sensitive_fields=["api_key", "sa_key"],
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
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers are: {supported}"
        )


def parse_provider_credentials(
    provider: str, credentials: object
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
