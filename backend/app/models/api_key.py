from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel

from app.core.util import now


class APIKeyBase(SQLModel):
    """Base model for API keys with foreign key fields."""

    # Foreign keys
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )
    user_id: int = Field(
        foreign_key="user.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={
            "comment": "Reference to the user for whom the API key was created"
        },
    )


class APIKeyPublic(APIKeyBase):
    id: UUID
    key_prefix: str  # Expose key_id for display (partial key identifier)
    inserted_at: datetime
    updated_at: datetime


class APIKeyCreateResponse(APIKeyPublic):
    """Response model when creating an API key includes the raw key only once"""

    key: str


class APIKeyVerifyResponse(SQLModel):
    """Response model for API key verification."""

    user_id: int
    organization_id: int
    project_id: int


class APIKey(APIKeyBase, table=True):
    """Database model for API keys."""

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the API key"},
    )
    key_prefix: str = Field(
        unique=True,
        index=True,
        nullable=False,
        sa_column_kwargs={
            "comment": "Unique prefix portion of the API key for identification"
        },
    )
    key_hash: str = Field(
        nullable=False,
        sa_column_kwargs={"comment": "Bcrypt hash of the secret of the API key"},
    )
    is_deleted: bool = Field(
        default=False,
        nullable=False,
        sa_column_kwargs={"comment": "Soft delete flag"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the API key was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the API key was last updated"},
    )
    deleted_at: datetime | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Timestamp when the API key was deleted"},
    )
