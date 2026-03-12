from datetime import datetime
from typing import Any, Literal

import sqlalchemy as sa
from app.core.util import now
from sqlmodel import Field, SQLModel
from sqlalchemy.dialects.postgresql import JSONB, ARRAY


class ModelConfigBase(SQLModel):
    provider: Literal["openai", "google"] = Field(
        default="openai",
        sa_column=sa.Column(
            sa.String, nullable=False, comment="provider name (e.g. openai, google)"
        ),
    )

    model_name: str = Field(
        ...,
        sa_column=sa.Column(
            sa.String,
            nullable=False,
            comment="model name (e.g. gpt-4o, gemini-3-flash-preview)",
        ),
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=sa.Column(JSONB, nullable=False, comment="model adhoc configuration"),
    )

    input_modalities: list[str] = Field(
        default_factory=list,
        sa_column=sa.Column(
            ARRAY(sa.String),
            nullable=False,
            server_default="{}",
            comment="supported input modalities: TEXT, IMAGE, PDF, AUDIO",
        ),
    )

    output_modalities: list[str] = Field(
        default_factory=list,
        sa_column=sa.Column(
            ARRAY(sa.String),
            nullable=False,
            server_default="{}",
            comment="supported output modalities: TEXT, AUDIO",
        ),
    )

    # NOTE: can we use this default_for column to help in routing?
    default_for: Literal["text", "stt", "tts"] | None = Field(
        default=None,
        sa_column=sa.Column(
            sa.String,
            nullable=True,
            comment=(
                "completion types this model is the default for. "
                "e.g. [text, stt, tts]. "
                "NULL means not a default. "
                "Supported: text, stt, tts"
            ),
        ),
    )

    is_active: bool = Field(
        default=True,
        sa_column=sa.Column(
            sa.Boolean,
            nullable=False,
            server_default=sa.text("true"),
            comment="whether this model is available",
        ),
    )


class ModelConfig(ModelConfigBase, table=True):
    __tablename__ = "model_config"
    __table_args__ = (
        sa.UniqueConstraint("provider", "model_name"),
        {"schema": "global"},
    )

    id: int | None = Field(
        default=None,
        sa_column=sa.Column(
            sa.Integer,
            primary_key=True,
            comment="unique identifier for model config table",
        ),
    )

    inserted_at: datetime = Field(
        default_factory=now,
        sa_column=sa.Column(
            sa.DateTime,
            default=now,
            nullable=False,
            comment="timestamp when model configuration was created",
        ),
    )

    updated_at: datetime = Field(
        default_factory=now,
        sa_column=sa.Column(
            sa.DateTime,
            default=now,
            nullable=False,
            onupdate=now,
            comment="timestamp when model configuration was updated",
        ),
    )


class ModelConfigPublic(ModelConfigBase):
    id: int
    inserted_at: datetime
    updated_at: datetime


class ModelConfigListPublic(SQLModel):
    data: list[ModelConfigPublic]
    count: int
