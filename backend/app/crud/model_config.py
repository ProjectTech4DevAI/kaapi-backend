from typing import Any, Literal

from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import sqltypes
from sqlmodel import Session, select

from app.models import ModelConfig

Provider = str
CompletionType = Literal["text", "stt", "tts"]


def list_active_model_configs(
    session: Session,
    provider: Provider | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[ModelConfig], bool]:
    statement = select(ModelConfig).where(ModelConfig.is_active)

    if provider:
        try:
            statement = statement.where(ModelConfig.provider == provider)
        except Exception:
            return [], False

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
        try:
            statement = statement.where(ModelConfig.provider == provider)
        except Exception:
            return []

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


def _modality_filter(stmt, completion_type: CompletionType):
    """Restrict query to models matching the completion type via modalities."""
    str_array = ARRAY(sqltypes.String)
    input_col = ModelConfig.input_modalities
    output_col = ModelConfig.output_modalities

    if completion_type == "stt":
        return stmt.where(
            input_col.cast(str_array).contains(["AUDIO"]),
            output_col.cast(str_array).contains(["TEXT"]),
        )
    if completion_type == "tts":
        return stmt.where(
            input_col.cast(str_array).contains(["TEXT"]),
            output_col.cast(str_array).contains(["AUDIO"]),
        )
    return stmt.where(
        input_col.cast(str_array).contains(["TEXT"]),
        output_col.cast(str_array).contains(["TEXT"]),
    )


def list_supported_models(
    session: Session, provider: Provider, completion_type: CompletionType
) -> list[str]:
    """Return active model names for a provider+completion type."""
    stmt = select(ModelConfig.model_name).where(
        ModelConfig.provider == provider,
        ModelConfig.is_active,
    )
    stmt = _modality_filter(stmt, completion_type)
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
        ModelConfig.is_active,
    )
    stmt = _modality_filter(stmt, completion_type)
    return session.exec(stmt).first() is not None


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
