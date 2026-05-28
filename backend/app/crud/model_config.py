from typing import Any, Literal

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models import ModelConfig
from app.models.llm.request import ConfigBlob
from app.models.model_config import CompletionType

Provider = Literal[
    "openai", "google", "sarvamai", "elevenlabs", "anthropic", "google-vertex"
]


def _normalize_provider(raw: str) -> str:
    """Map NativeCompletionConfig providers (e.g. 'openai-native') to model_config provider names."""
    return raw[: -len("-native")] if raw.endswith("-native") else raw


def list_active_model_configs(
    session: Session,
    provider: Provider | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[ModelConfig], bool]:
    statement = select(ModelConfig).where(ModelConfig.is_active)

    if provider:
        statement = statement.where(ModelConfig.provider == provider)

    statement = statement.order_by(ModelConfig.provider, ModelConfig.model_name)
    statement = statement.offset(skip).limit(limit + 1)
    models = list(session.exec(statement).all())

    has_more = False
    if len(models) > limit:
        has_more = True
        models = models[:limit]

    return models, has_more


def list_all_active_model_configs(
    session: Session,
    provider: Provider | None = None,
) -> list[ModelConfig]:
    statement = select(ModelConfig).where(ModelConfig.is_active)

    if provider:
        statement = statement.where(ModelConfig.provider == provider)

    statement = statement.order_by(ModelConfig.provider, ModelConfig.model_name)
    return list(session.exec(statement).all())


def get_model_config(
    session: Session, provider: Provider, model_name: str
) -> ModelConfig | None:
    statement = select(ModelConfig).where(
        ModelConfig.provider == provider,
        ModelConfig.model_name == model_name,
        ModelConfig.is_active,
    )
    return session.exec(statement).first()


def list_supported_models(
    session: Session, provider: Provider, completion_type: CompletionType
) -> list[str]:
    """Return active model names for a provider + completion type."""
    stmt = select(ModelConfig.model_name).where(
        ModelConfig.provider == provider,
        ModelConfig.completion_type == completion_type,
        ModelConfig.is_active,
    )
    return list(session.exec(stmt).all())


def is_model_supported(
    session: Session,
    provider: Provider,
    completion_type: CompletionType,
    model_name: str,
) -> bool:
    """Check whether (provider, model_name) is active and matches the completion type."""
    stmt = select(ModelConfig.id).where(
        ModelConfig.provider == provider,
        ModelConfig.model_name == model_name,
        ModelConfig.completion_type == completion_type,
        ModelConfig.is_active,
    )
    return session.exec(stmt).first() is not None


def validate_blob_model_or_raise(session: Session, blob: ConfigBlob) -> None:
    """Reject ConfigBlob whose completion.params.model is not in model_config.

    model_config is the source of truth — all providers/types validated.
    Native configs are exempt (they forward raw params to the provider).
    """
    completion = blob.completion
    raw_provider = completion.provider
    completion_type = completion.type
    if raw_provider is None:
        return

    if raw_provider.endswith("-native"):
        return

    provider = _normalize_provider(raw_provider)

    model_name = (completion.params or {}).get("model")
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail=f"completion.params.model is required for provider='{raw_provider}'",
        )

    model_row = get_model_config(
        session=session,
        provider=provider,  # type: ignore[arg-type]
        model_name=model_name,
    )
    if model_row is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found for provider='{provider}'.",
        )

    if not is_model_supported(
        session=session,
        provider=provider,  # type: ignore[arg-type]
        completion_type=completion_type,
        model_name=model_name,
    ):
        allowed = list_supported_models(
            session=session,
            provider=provider,  # type: ignore[arg-type]
            completion_type=completion_type,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' is not supported for provider='{provider}' "
                f"type='{completion_type}'. Allowed: {allowed}"
            ),
        )

    if completion_type == "tts":
        voice = (completion.params or {}).get("voice")
        voice_spec = (
            model_row.config.get("voice")
            if isinstance(model_row.config, dict)
            else None
        )
        allowed_voices = (
            voice_spec.get("options") if isinstance(voice_spec, dict) else None
        )
        if voice and allowed_voices and voice not in allowed_voices:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Voice '{voice}' is not supported for provider='{provider}' "
                    f"model='{model_name}'. Allowed: {allowed_voices}"
                ),
            )


def is_reasoning_model(session: Session, provider: Provider, model_name: str) -> bool:
    """Return True if the model is configured with a reasoning `effort` control.

    A model is considered reasoning-capable if its `config` JSON contains an
    `effort` key; models that instead expose a `temperature` key are treated
    as standard chat models.
    """
    model = get_model_config(session=session, provider=provider, model_name=model_name)
    if model is None or not isinstance(model.config, dict):
        return False
    return "effort" in model.config


def estimate_model_cost(
    session: Session,
    provider: Provider,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    usage_type: Literal["response", "batch"] = "response",
) -> dict[str, Any] | None:
    model = get_model_config(session=session, provider=provider, model_name=model_name)
    if model is None or model.pricing is None:
        return None

    if not isinstance(model.pricing, dict):
        return None

    pricing_source: dict[str, Any] = model.pricing
    usage_pricing = pricing_source.get(usage_type)
    if not isinstance(usage_pricing, dict):
        return None

    input_price = usage_pricing.get("input_token_cost")
    output_price = usage_pricing.get("output_token_cost")

    if not isinstance(input_price, (int, float)) or not isinstance(
        output_price, (int, float)
    ):
        return None

    input_cost = (input_tokens / 1_000_000) * float(input_price)
    output_cost = (output_tokens / 1_000_000) * float(output_price)

    return {
        "provider": provider,
        "model_name": model_name,
        "usage_type": usage_type,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "currency": "USD",
    }
