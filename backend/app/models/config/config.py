from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

import sqlalchemy as sa
from pydantic import field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import event
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm.base import NO_VALUE
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


@event.listens_for(Config.tag, "set", active_history=True)
def _enforce_tag_immutable(
    _target: "Config", value: ConfigTag, oldvalue: ConfigTag, _initiator: object
) -> None:
    """`tag` is fixed at creation — it selects the config_blob shape, so changing
    it would leave the blob in the wrong shape. Allow the initial assignment and
    no-op sets; reject any real change on a persisted row. ``active_history`` forces
    the loaded value to be available as ``oldvalue`` so the comparison is reliable.
    """
    if oldvalue not in (NO_VALUE, None) and value != oldvalue:
        raise ValueError("config tag is immutable and cannot be changed after creation")


class ConfigCreate(ConfigBase):
    """Create new configuration"""

    tag: ConfigTag = Field(
        default=ConfigTag.DEFAULT,
        description=(
            "Optional tag for classifying this config. Omit to store 'default'; "
            "set 'ASSESSMENT' for assessment use."
        ),
    )
    config_blob: ConfigBlob | SkipJsonSchema[AssessmentConfigBlob] = Field(
        description="Provider-specific parameters; shape must match `tag`"
    )
    commit_message: str | None = Field(
        default=None,
        max_length=512,
        description="Optional message describing the changes in this version",
    )

    @field_validator("config_blob", mode="before")
    @classmethod
    def _validate_blob_for_tag(cls, blob: object, info: object) -> object:
        """Validate config_blob against ONLY the model its tag dictates, so a
        `default` config never surfaces assessment-branch errors and vice-versa.

        Without this, `config_blob` is a `ConfigBlob | AssessmentConfigBlob`
        union: a blob that is invalid for its tag fails one branch and pydantic
        leaks the *other* branch's error (e.g. a bad `default` completion
        reporting "assessment: Field required"). Runs before union coercion; raw
        JSON bodies (dicts) only — model instances built in code fall through."""
        if not isinstance(blob, dict):
            return blob
        raw_tag = info.data.get("tag", ConfigTag.DEFAULT)
        tag = raw_tag.value if isinstance(raw_tag, ConfigTag) else raw_tag
        is_assessment = tag == ConfigTag.ASSESSMENT.value

        if is_assessment and "assessment" not in blob:
            raise ValueError(
                "An ASSESSMENT config requires an 'assessment' block in config_blob."
            )
        if not is_assessment and "completion" not in blob:
            raise ValueError(
                "A default config requires a 'completion' block in config_blob."
            )

        # Validate against the single tag-appropriate model and return the
        # instance; its ValidationError (correct branch only) propagates as-is.
        model = AssessmentConfigBlob if is_assessment else ConfigBlob
        return model.model_validate(blob)

    @model_validator(mode="after")
    def _check_blob_matches_tag(self) -> "ConfigCreate":
        expected = (
            AssessmentConfigBlob if self.tag == ConfigTag.ASSESSMENT else ConfigBlob
        )
        if not isinstance(self.config_blob, expected):
            raise ValueError(f"config_blob shape does not match tag '{self.tag.value}'")
        return self


class ConfigUpdate(SQLModel):
    # Only `name` and `description` are updatable. `extra="forbid"` rejects any other
    # field (e.g. `tag`, `config_blob`) with a 422 instead of silently ignoring it —
    # `tag` is fixed at creation (it selects the config_blob shape) and `config_blob`
    # is changed only through versioning.
    model_config = {"extra": "forbid"}

    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(
        default=None, max_length=512, description="Optional description"
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
