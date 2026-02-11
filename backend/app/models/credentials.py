from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlmodel import Field, Relationship, SQLModel

from app.core.util import now
from app.models.organization import Organization
from app.models.project import Project


class CredsBase(SQLModel):
    """Base model for credentials with foreign keys and common fields."""

    is_active: bool = Field(
        default=True,
        nullable=False,
        sa_column_kwargs={
            "comment": "Flag indicating if this credential is currently active and usable"
        },
    )

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


class CredsCreate(SQLModel):
    """Create new credentials for an organization.
    The credential field should be a dictionary mapping provider names to their credentials.
    Example: {"openai": {"api_key": "..."}, "langfuse": {"public_key": "..."}}
    """

    is_active: bool = True
    credential: dict[str, Any] = Field(
        default=None,
        description="Dictionary mapping provider names to their credentials",
    )


class CredsUpdate(SQLModel):
    """Update credentials for an organization.
    Can update a specific provider's credentials or add a new provider.
    """

    provider: str = Field(
        description="Name of the provider to update/add credentials for"
    )
    credential: dict[str, Any] = Field(
        description="Credentials for the specified provider",
    )
    is_active: bool | None = Field(
        default=None, description="Whether the credentials are active"
    )


class Credential(CredsBase, table=True):
    """Database model for storing provider credentials.
    Each row represents credentials for a single provider.
    """

    __table_args__ = (
        sa.UniqueConstraint(
            "organization_id",
            "project_id",
            "provider",
            name="uq_credential_org_project_provider",
        ),
    )

    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique ID for the credential"},
    )
    provider: str = Field(
        index=True,
        nullable=False,
        description="Provider name like 'openai', 'google'",
        sa_column_kwargs={"comment": "Provider name like 'openai', 'google'"},
    )
    credential: str = Field(
        nullable=False,
        description="Encrypted JSON string containing provider-specific API credentials",
        sa_column_kwargs={
            "comment": "Encrypted JSON string containing provider-specific API credentials"
        },
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the credential was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the credential was last updated"},
    )

    # Relationships
    organization: Organization | None = Relationship(back_populates="creds")
    project: Project | None = Relationship(back_populates="creds")

    def to_public(self) -> "CredsPublic":
        """Convert the database model to a public model with decrypted credentials."""
        from app.core.security import decrypt_credentials

        return CredsPublic(
            id=self.id,
            organization_id=self.organization_id,
            project_id=self.project_id,
            is_active=self.is_active,
            provider=self.provider,
            credential=decrypt_credentials(self.credential)
            if self.credential
            else None,
            inserted_at=self.inserted_at,
            updated_at=self.updated_at,
        )


class CredsPublic(CredsBase):
    """Public representation of credentials, excluding sensitive information."""

    id: int
    provider: str
    credential: dict[str, Any] | None = None
    inserted_at: datetime
    updated_at: datetime
