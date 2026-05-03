from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlmodel import Field, SQLModel, Index, text
from pydantic import field_validator

from app.core.util import now
from app.models.llm.request import ConfigBlob
from .version import ConfigVersionPublic


class ConfigTag(str, Enum):
    """Config classification tag."""

    DEFAULT = "default"
    ASSESSMENT = "ASSESSMENT"


_CONFIG_TAG_PG_ENUM = postgresql.ENUM(
    ConfigTag,
    name="config_tag",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
    create_type=False,
)


class ConfigBase(SQLModel):
    """Base model for LLM configuration metadata"""

    name: str = Field(
        min_length=1,
        max_length=128,
        description="Config name",
        sa_column_kwargs={"comment": "Configuration name"},
    )
    description: str | None = Field(
        default=None,
        max_length=512,
        description="Description of the configuration",
        sa_column_kwargs={"comment": "Description of the configuration"},
    )


class Config(ConfigBase, table=True):
    """Database model for LLM configuration storage"""

    __tablename__ = "config"
    __table_args__ = (
        Index(
            "uq_config_project_id_name_active",
            "project_id",
            "name",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index(
            "idx_config_project_id_updated_at_active",
            "project_id",
            "updated_at",
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the configuration"},
    )

    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )

    tag: ConfigTag | None = Field(
        default=None,
        sa_column=sa.Column(
            _CONFIG_TAG_PG_ENUM,
            nullable=True,
            comment=(
                "Optional tag classifying the config: 'default' for general use, "
                "'ASSESSMENT' for assessment UI picker; NULL means untagged."
            ),
        ),
    )

    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the configuration was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the configuration was last updated"
        },
    )

    deleted_at: datetime | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Timestamp when the configuration was deleted"},
    )


class ConfigCreate(ConfigBase):
    """Create new configuration"""

    # Initial version data
    config_blob: ConfigBlob = Field(description="Provider-specific parameters")
    commit_message: str | None = Field(
        default=None,
        max_length=512,
        description="Optional message describing the changes in this version",
    )
    tag: ConfigTag | None = Field(
        default=None,
        description=(
            "Optional tag for classifying this config. "
            "Omit (NULL) for untagged/general-purpose; set 'ASSESSMENT' to surface "
            "the config in the Assessment UI picker."
        ),
    )

    @field_validator("config_blob")
    def validate_blob_not_empty(cls, value):
        if not value:
            raise ValueError("config_blob cannot be empty")
        return value


class ConfigUpdate(SQLModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(
        default=None, max_length=512, description="Optional description"
    )
    tag: ConfigTag | None = Field(
        default=None,
        description=(
            "Optional tag for classifying this config. "
            "Currently set-only; omitting it leaves the existing tag untouched."
        ),
    )


class ConfigPublic(ConfigBase):
    id: UUID
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class ConfigWithVersion(ConfigPublic):
    version: ConfigVersionPublic


class ConfigWithVersions(ConfigPublic):
    versions: list[ConfigVersionPublic]
