import logging
from datetime import datetime
from typing import Any, Literal, get_args

from fastapi import HTTPException
from pydantic import JsonValue, ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.models import ModelConfig
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.models.llm.constants import CompletionType
from app.models.llm.constants import Provider as ProviderEnum
from app.models.llm.request import CompletionConfig, ConfigBlob, KaapiLLMParams
from app.models.model_config import (
    ModelConfigBulkUpdateItem,
    ModelConfigCreate,
    ModelConfigUpdate,
)

logger = logging.getLogger(__name__)

Provider = Literal[
    "openai",
    "google",
    "sarvamai",
    "elevenlabs",
    "anthropic",
    "google-aistudio",
    "proxy",
]

# Runtime view of the Provider Literal. Use this anywhere the `global.provider_enum`
# values are needed (filter validation, cost-lookup guards) so the set stays in sync
# with the Literal definition.
KNOWN_PROVIDERS: frozenset[str] = frozenset(get_args(Provider))

# Suffix that distinguishes a NativeCompletionConfig provider (e.g. "openai-native")
# from the canonical provider name stored in model_config ("openai").
NATIVE_PROVIDER_SUFFIX = "-native"


def _normalize_provider(raw: str) -> str:
    """Map NativeCompletionConfig providers (e.g. 'openai-native') to model_config provider names."""
    return (
        raw[: -len(NATIVE_PROVIDER_SUFFIX)]
        if raw.endswith(NATIVE_PROVIDER_SUFFIX)
        else raw
    )


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
    session: Session,
    provider: Provider,
    model_name: str,
    include_inactive: bool = False,
) -> ModelConfig | None:
    statement = select(ModelConfig).where(
        ModelConfig.provider == provider,
        ModelConfig.model_name == model_name,
    )
    if not include_inactive:
        statement = statement.where(ModelConfig.is_active)
    return session.exec(statement).first()


def list_supported_models(
    session: Session, provider: Provider, completion_type: CompletionType
) -> list[str]:
    """Return active model names for a provider + completion type."""
    stmt = select(ModelConfig.model_name).where(
        ModelConfig.provider == provider,
        ModelConfig.completion_type.contains([completion_type]),  # type: ignore[union-attr]
        ModelConfig.is_active,
    )
    return list(session.exec(stmt).all())


def is_model_supported(
    session: Session,
    provider: Provider,
    completion_type: CompletionType,
    model_name: str,
) -> bool:
    """Check whether (provider, model_name) is active and supports the completion type."""
    stmt = select(ModelConfig.id).where(
        ModelConfig.provider == provider,
        ModelConfig.model_name == model_name,
        ModelConfig.completion_type.contains([completion_type]),  # type: ignore[union-attr]
        ModelConfig.is_active,
    )
    return session.exec(stmt).first() is not None


def validate_blob_model_or_raise(session: Session, blob: ConfigBlob) -> None:
    """Reject ConfigBlob whose completion.params.model is not in model_config."""
    _validate_completion_model_or_raise(session, blob.completion)


def validate_config_blob_for_tag(
    session: Session, tag: ConfigTag, raw_blob: dict[str, JsonValue]
) -> ConfigBlob | AssessmentConfigBlob:
    """Validate a raw (or merged partial) blob against the type its tag dictates.

    DEFAULT → ConfigBlob, ASSESSMENT → AssessmentConfigBlob. Shape errors surface
    as 422; the model-existence check stays liberal (warn, don't raise).
    """
    blob_type: type[ConfigBlob] | type[AssessmentConfigBlob] = (
        AssessmentConfigBlob if tag == ConfigTag.ASSESSMENT else ConfigBlob
    )
    try:
        blob = blob_type.model_validate(raw_blob)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    validate_blob_completion_models(session, blob)
    return blob


def validate_blob_completion_models(
    session: Session, blob: ConfigBlob | AssessmentConfigBlob
) -> None:
    """Run the model-existence check on an already-parsed blob (create path).

    For an assessment blob every present pre-filter runs its own completion, so
    each is validated on the same path as the main completion.
    """
    completion = (
        blob.assessment if isinstance(blob, AssessmentConfigBlob) else blob.completion
    )
    _validate_completion_model_or_raise(session, completion)

    if isinstance(blob, AssessmentConfigBlob) and blob.pre_filters is not None:
        pre_filters = (
            blob.pre_filters.topic_relevance,
            blob.pre_filters.duplicate_detection,
        )
        for flt in pre_filters:
            if flt is not None:
                _validate_model_or_raise(
                    session,
                    raw_provider=flt.provider,
                    completion_type=CompletionType.TEXT,
                    params=flt.params,
                )


def _validate_completion_model_or_raise(
    session: Session, completion: CompletionConfig
) -> None:
    """Reject a completion whose params.model is not in model_config."""
    _validate_model_or_raise(
        session,
        raw_provider=completion.provider,
        completion_type=completion.type,
        params=completion.params,
    )


def _get_param(params: KaapiLLMParams | dict[str, Any] | None, key: str) -> Any:
    """Read a field from completion params, dict or typed Kaapi model alike."""
    if isinstance(params, dict):
        return params.get(key)
    return getattr(params, key, None)


def _validate_model_or_raise(
    session: Session,
    *,
    raw_provider: str | None,
    completion_type: str,
    params: KaapiLLMParams | dict[str, Any] | None,
) -> None:
    """Reject a (provider, type, params) whose params.model is not in model_config.

    model_config is the source of truth — all providers/types validated.
    Native configs are exempt (they forward raw params to the provider).

    # As of now - this whole validation is liberal
    # change this if we want to be more strict about unsupported models/providers or missing model configs.
    """

    # Proxy forwards the request to the client's own LLM endpoint — no model
    # lookup, no provider mapping.
    if completion_type == ProviderEnum.PROXY.value:
        return

    if raw_provider is None:
        return

    if raw_provider.endswith(NATIVE_PROVIDER_SUFFIX):
        return

    provider = _normalize_provider(raw_provider)

    model_name = _get_param(params, "model") or None
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
        logger.warning(
            f"[_validate_model_or_raise] Model '{model_name}' not found for provider='{provider}'."
            "Kaapi does not yet support this model, but will forward as long as the `model` field has no typos and the model is not deprecated by the provider"
        )

    if completion_type == "tts" and model_row is not None:
        voice = _get_param(params, "voice")
        voice_spec = (
            model_row.config.get("voice")
            if isinstance(model_row.config, dict)
            else None
        )
        allowed_voices = (
            voice_spec.get("options") if isinstance(voice_spec, dict) else None
        )
        if voice and allowed_voices and voice not in allowed_voices:
            logger.warning(
                f"[_validate_model_or_raise] Voice '{voice}' is not supported for provider='{provider}' "
                f"model='{model_name}'. Allowed: {allowed_voices}."
            )


def bulk_create_model_configs(
    session: Session, items: list[ModelConfigCreate]
) -> list[ModelConfig]:
    models = [ModelConfig.model_validate(item) for item in items]
    session.add_all(models)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="Duplicate (provider, model_name) — entry already exists",
        ) from e
    for m in models:
        session.refresh(m)
    return models


def update_model_config(
    session: Session, provider: str, model_name: str, data: ModelConfigUpdate
) -> ModelConfig:
    model = get_model_config(
        session=session,
        provider=provider,  # type: ignore[arg-type]
        model_name=model_name,
        include_inactive=True,
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)
    model.updated_at = datetime.utcnow()
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


def bulk_update_model_configs(
    session: Session, items: list[ModelConfigBulkUpdateItem]
) -> list[ModelConfig]:
    keys = [(item.provider, item.model_name) for item in items]
    existing: dict[tuple, ModelConfig] = {}
    for provider, model_name in keys:
        m = get_model_config(
            session=session,
            provider=provider,
            model_name=model_name,
            include_inactive=True,
        )
        if m is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found for provider='{provider}'",
            )
        existing[(provider, model_name)] = m
    updated = []
    now = datetime.utcnow()
    for item in items:
        model = existing[(item.provider, item.model_name)]
        for field, value in item.model_dump(
            exclude_unset=True, exclude={"provider", "model_name"}
        ).items():
            setattr(model, field, value)
        model.updated_at = now
        session.add(model)
        updated.append(model)
    session.commit()
    for m in updated:
        session.refresh(m)
    return updated


def delete_model_config(session: Session, provider: str, model_name: str) -> None:
    model = get_model_config(
        session=session,
        provider=provider,  # type: ignore[arg-type]
        model_name=model_name,
        include_inactive=True,
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    session.delete(model)
    session.commit()


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


def is_summary_model(session: Session, provider: Provider, model_name: str) -> bool:
    """Return True if the model is configured with a summary `effort` control.

    A model is considered summary-capable if its `config` JSON contains an
    `summary` key; models that instead expose a `temperature` key are treated
    as standard chat models.
    """
    model = get_model_config(session=session, provider=provider, model_name=model_name)
    if model is None or not isinstance(model.config, dict):
        return False
    return "summary" in model.config


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
