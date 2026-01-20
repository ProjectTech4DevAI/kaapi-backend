from datetime import datetime
from typing import TYPE_CHECKING

from sqlmodel import Field, Relationship, SQLModel

from app.core.util import now

if TYPE_CHECKING:
    from .assistants import Assistant
    from .collection import Collection
    from .credentials import Credential
    from .openai_conversation import OpenAIConversation
    from .project import Project


# Shared properties for an Organization
class OrganizationBase(SQLModel):
    """Base model for organizations with common data fields."""

    name: str = Field(
        unique=True,
        index=True,
        max_length=255,
        sa_column_kwargs={"comment": "Organization name (unique identifier)"},
    )
    is_active: bool = Field(
        default=True,
        sa_column_kwargs={"comment": "Flag indicating if the organization is active"},
    )


# Properties to receive via API on creation
class OrganizationCreate(OrganizationBase):
    pass


# Properties to receive via API on update, all are optional
class OrganizationUpdate(SQLModel):
    name: str | None = Field(default=None, max_length=255)
    is_active: bool | None = Field(default=None)


# Database model for Organization
class Organization(OrganizationBase, table=True):
    """Database model for organizations."""

    id: int = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the organization"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the organization was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the organization was last updated"
        },
    )

    # Relationships
    creds: list["Credential"] = Relationship(
        back_populates="organization", cascade_delete=True
    )
    project: list["Project"] = Relationship(
        back_populates="organization", cascade_delete=True
    )
    assistants: list["Assistant"] = Relationship(
        back_populates="organization", cascade_delete=True
    )
    openai_conversations: list["OpenAIConversation"] = Relationship(
        back_populates="organization", cascade_delete=True
    )


# Properties to return via API
class OrganizationPublic(OrganizationBase):
    id: int
    inserted_at: datetime
    updated_at: datetime


class OrganizationsPublic(SQLModel):
    data: list[OrganizationPublic]
    count: int
