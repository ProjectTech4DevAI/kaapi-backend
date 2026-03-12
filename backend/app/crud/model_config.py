import logging
from typing import Optional, Literal

from sqlmodel import Session, select

from app.models import ModelConfig

logger = logging.getLogger(__name__)


def get_default_model_for_type(
    session: Session, completion_type: Literal["text", "stt", "tts"]
) -> Optional[ModelConfig]:
    statement = (
        select(ModelConfig)
        .where(
            ModelConfig.is_active == True,
            ModelConfig.default_for == completion_type,
        )
        .limit(1)
    )

    return session.exec(statement).first()


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
