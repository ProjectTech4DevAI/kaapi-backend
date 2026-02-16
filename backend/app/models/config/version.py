from datetime import datetime
from uuid import UUID, uuid4
from typing import Any

from pydantic import field_validator
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel, UniqueConstraint, Index, text

from app.core.util import now
from app.models.llm.request import ConfigBlob


class ConfigVersionBase(SQLModel):
    config_blob: dict[str, Any] = Field(
        sa_column=sa.Column(
            JSONB,
            nullable=False,
            comment="Provider-specific configuration parameters (temperature, max_tokens, etc.)",
        ),
        description="Provider-specific configuration parameters (temperature, max_tokens, etc.)",
    )
    commit_message: str | None = Field(
        default=None,
        max_length=512,
        description="Optional message describing the changes in this version",
        sa_column_kwargs={
            "comment": "Optional message describing the changes in this version"
        },
    )

    @field_validator("config_blob")
    def validate_blob_not_empty(cls, value):
        if not value:
            raise ValueError("config_blob cannot be empty")
        return value


class ConfigVersion(ConfigVersionBase, table=True):
    __tablename__ = "config_version"
    __table_args__ = (
        UniqueConstraint(
            "config_id", "version", name="uq_config_version_config_id_version"
        ),
        Index(
            "idx_config_version_config_id_version_active",
            "config_id",
            "version",
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the configuration version"},
    )

    config_id: UUID = Field(
        foreign_key="config.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the parent configuration"},
    )
    version: int = Field(
        nullable=False,
        description="Version number starting at 1",
        ge=1,
        sa_column_kwargs={"comment": "Version number starting at 1"},
    )

    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the version was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the version was last updated"},
    )

    deleted_at: datetime | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Timestamp when the version was soft-deleted"},
    )


class ConfigVersionCreate(ConfigVersionBase):
    # Store config_blob as JSON in the DB. Validation uses ConfigBlob only at creation
    # time, since schema may evolve. When fetching, it is returned as a raw dict and
    # re-validated against the latest schema before use.
    config_blob: ConfigBlob = Field(
        description="Provider-specific configuration parameters (temperature, max_tokens, etc.)",
    )


class ConfigVersionUpdate(SQLModel):
    """
    Partial update model for creating a new config version.

    Only the fields that need to change should be provided.
    Fields like 'type'(text, stt,tts) are inherited from the existing config
    and cannot be changed,
    """

    config_blob: dict[str, Any] = Field(
        description="Partial config blob. Only include fields you want to update. "
        "Provider and type are inherited from existing config and cannot be changed.",
    )
    commit_message: str | None = Field(
        default=None,
        max_length=512,
        description="Optional message describing the changes in this version",
    )


class ConfigVersionPublic(ConfigVersionBase):
    id: UUID = Field(description="Unique id for the configuration version")
    config_id: UUID = Field(description="Id of the parent configuration")
    version: int = Field(nullable=False, description="Version number starting at 1")
    inserted_at: datetime
    updated_at: datetime


class ConfigVersionItems(SQLModel):
    """Lightweight version for lists (without large config_blob)"""

    id: UUID = Field(description="Unique id for the configuration version")
    version: int = Field(nullable=False, description="Version number starting at 1")
    config_id: UUID = Field(description="Id of the parent configuration")
    commit_message: str | None = Field(
        default=None,
        max_length=512,
        description="Optional message describing the changes in this version",
    )
    inserted_at: datetime
    updated_at: datetime
