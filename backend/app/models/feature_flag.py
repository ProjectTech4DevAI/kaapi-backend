from datetime import datetime

from sqlmodel import Field, SQLModel, UniqueConstraint

from app.core.util import now


class FeatureFlagBase(SQLModel):
    key: str = Field(
        min_length=1,
        max_length=128,
        index=True,
        sa_column_kwargs={"comment": "Feature flag key (matches FeatureFlag enum value)"},
    )
    organization_id: int = Field(
        foreign_key="organization.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Organization scope for this feature flag"},
    )
    project_id: int | None = Field(
        default=None,
        foreign_key="project.id",
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Optional project scope for this feature flag"},
    )
    enabled: bool = Field(
        sa_column_kwargs={"comment": "Whether the feature flag is enabled"},
    )


class FeatureFlag(FeatureFlagBase, table=True):
    __tablename__ = "feature_flag"
    __table_args__ = (
        UniqueConstraint(
            "key",
            "organization_id",
            "project_id",
            name="uq_feature_flag_key_org_project",
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
    key: str = Field(min_length=1, max_length=128)
    organization_id: int
    project_id: int | None = None
    enabled: bool


class FeatureFlagUpdate(SQLModel):
    key: str = Field(min_length=1, max_length=128)
    organization_id: int
    project_id: int | None = None
    enabled: bool


class FeatureFlagDelete(SQLModel):
    key: str = Field(min_length=1, max_length=128)
    organization_id: int
    project_id: int | None = None


class FeatureFlagPublic(FeatureFlagBase):
    id: int
    inserted_at: datetime
    updated_at: datetime
