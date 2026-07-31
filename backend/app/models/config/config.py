from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

import sqlalchemy as sa
from pydantic import model_validator
from sqlalchemy.dialects import postgresql
from sqlmodel import Field, Index, SQLModel, text

from app.core.util import now
from app.models.llm.request import ConfigBlob

from .assessment_blob import AssessmentConfigBlob
from .version import ConfigVersionPublic


class ConfigTag(StrEnum):
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
        Index(
            "idx_config_project_id_tag_active",
            "project_id",
            "tag",
            text("updated_at DESC"),
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

    tag: ConfigTag = Field(
        default=ConfigTag.DEFAULT,
        sa_column=sa.Column(
            _CONFIG_TAG_PG_ENUM,
            nullable=False,
            server_default=sa.text("'default'::config_tag"),
            comment=(
                "Tag classifying the config: 'default' for general use, "
                "'ASSESSMENT' for assessment use."
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

    # Shape picked by `tag`; `_check_blob_matches_tag` enforces the pairing.
    config_blob: ConfigBlob | AssessmentConfigBlob = Field(
        description="Provider-specific parameters; shape must match `tag`"
    )
    commit_message: str | None = Field(
        default=None,
        max_length=512,
        description="Optional message describing the changes in this version",
    )
    tag: ConfigTag = Field(
        default=ConfigTag.DEFAULT,
        description=(
            "Optional tag for classifying this config. Omit to store 'default'; "
            "set 'ASSESSMENT' for assessment use."
        ),
    )

    @model_validator(mode="after")
    def _check_blob_matches_tag(self) -> "ConfigCreate":
        expected = (
            AssessmentConfigBlob if self.tag == ConfigTag.ASSESSMENT else ConfigBlob
        )
        if not isinstance(self.config_blob, expected):
            raise ValueError(
                f"config_blob shape does not match tag '{self.tag.value}'"
            )
        return self


class ConfigUpdate(SQLModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(
        default=None, max_length=512, description="Optional description"
    )
    tag: ConfigTag | None = Field(
        default=None,
        description=("Optional tag for classifying this config. "),
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
