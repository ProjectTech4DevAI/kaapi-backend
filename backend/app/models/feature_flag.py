from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Index
from sqlmodel import Field, SQLModel

from app.core.feature_flags.constants import FeatureFlag as FeatureFlagKeyEnum
from app.core.util import now


class FeatureFlagBase(SQLModel):
    key: FeatureFlagKeyEnum = Field(
        sa_column=sa.Column(
            sa.Enum(FeatureFlagKeyEnum, name="featureflagkey"),
            nullable=False,
            index=True,
            comment="Feature flag key",
        ),
    )
    organization_id: int = Field(
        foreign_key="organization.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Organization scope for this feature flag"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Project scope for this feature flag"},
    )
    enabled: bool = Field(
        sa_column_kwargs={"comment": "Whether the feature flag is enabled"},
    )


class FeatureFlag(FeatureFlagBase, table=True):
    __tablename__ = "feature_flag"
    __table_args__ = (
        Index(
            "uq_feature_flag_key_org_project",
            "key",
            "organization_id",
            "project_id",
            unique=True,
        ),
    )

    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for feature flag"},
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the flag row was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the flag row was last updated"},
    )


class FeatureFlagCreate(SQLModel):
    key: FeatureFlagKeyEnum
    project_id: int
    enabled: bool

    model_config = {
        "json_schema_extra": {
            "examples": [{"key": "ASSESSMENT", "project_id": 1, "enabled": True}]
        }
    }


class FeatureFlagUpdate(FeatureFlagCreate):
    """Same fields as FeatureFlagCreate; distinct name for OpenAPI schema."""

    model_config = {
        "json_schema_extra": {
            "examples": [{"key": "ASSESSMENT", "project_id": 1, "enabled": False}]
        }
    }


class FeatureFlagDelete(SQLModel):
    key: FeatureFlagKeyEnum
    project_id: int

    model_config = {
        "json_schema_extra": {"examples": [{"key": "ASSESSMENT", "project_id": 1}]}
    }


class FeatureFlagPublic(FeatureFlagBase):
    id: int
    inserted_at: datetime
    updated_at: datetime
