from datetime import datetime
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlmodel import Field, SQLModel

from app.core.util import now


class ModelConfigBase(SQLModel):
    provider: Literal["openai", "google", "sarvamai", "elevenlabs"] = Field(
        default="openai",
        sa_column=sa.Column(
            sa.String,
            nullable=False,
            comment="provider name (e.g. openai, google, sarvamai, elevenlabs)",
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
            comment="supported input modalities: TEXT, IMAGE, FILES, AUDIO",
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

    pricing: dict[str, Any] | None = Field(
        default=None,
        sa_column=sa.Column(
            JSONB,
            nullable=True,
            comment=(
                "pricing per 1M tokens in USD. "
                "Structure: {response: {input_token_cost, output_token_cost}, "
                "batch: {input_token_cost, output_token_cost}}"
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
