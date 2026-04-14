import logging
from typing import Any, Optional, Literal

from sqlmodel import Session, select

from app.models import ModelConfig

logger = logging.getLogger(__name__)


def get_active_models(
    session: Session,
    provider: Literal["openai", "google"] | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[ModelConfig]:
    statement = select(ModelConfig).where(ModelConfig.is_active == True)

    if provider:
        statement = statement.where(ModelConfig.provider == provider)

    statement = statement.order_by(ModelConfig.provider, ModelConfig.model_name)
    statement = statement.offset(skip).limit(limit)
    return list(session.exec(statement).all())


def get_model_config(
    session: Session, provider: Literal["openai", "google"], model_name: str
) -> Optional[ModelConfig]:
    statement = select(ModelConfig).where(
        ModelConfig.provider == provider,
        ModelConfig.model_name == model_name,
        ModelConfig.is_active == True,
    )
    return session.exec(statement).first()


def estimate_model_cost(
    session: Session,
    provider: Literal["openai", "google"],
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    tag: Literal["response", "batch"] = "response",
) -> Optional[dict[str, Any]]:
    model = get_model_config(session=session, provider=provider, model_name=model_name)
    if model is None or model.pricing is None:
        return None

    if not isinstance(model.pricing, dict):
        return None

    pricing_source: dict[str, Any] = model.pricing
    tag_pricing = pricing_source.get(tag)
    if not isinstance(tag_pricing, dict):
        return None

    input_price = tag_pricing.get("input_token_cost")
    output_price = tag_pricing.get("output_token_cost")

    if not isinstance(input_price, (int, float)) or not isinstance(
        output_price, (int, float)
    ):
        return None

    input_cost = (input_tokens / 1_000_000) * float(input_price)
    output_cost = (output_tokens / 1_000_000) * float(output_price)
    total_cost = round(input_cost + output_cost, 4)

    return {
        "provider": provider,
        "model_name": model_name,
        "tag": tag,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "currency": "USD",
    }
