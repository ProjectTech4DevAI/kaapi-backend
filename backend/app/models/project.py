from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID, uuid4

from pydantic import JsonValue
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint

from app.core.util import now

if TYPE_CHECKING:
    from .assistants import Assistant
    from .collection import Collection
    from .credentials import Credential
    from .fine_tuning import FineTuning
    from .openai_conversation import OpenAIConversation
    from .organization import Organization


# Shared properties for a Project
class ProjectBase(SQLModel):
    """Base model for projects with common data fields."""

    name: str = Field(
        index=True,
        max_length=255,
        sa_column_kwargs={"comment": "Project name"},
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        sa_column_kwargs={"comment": "Project description"},
    )
    is_active: bool = Field(
        default=True,
        sa_column_kwargs={"comment": "Flag indicating if the project is active"},
    )


# Properties to receive via API on creation
class ProjectCreate(ProjectBase):
    organization_id: int


# Properties to receive via API on update, all are optional
class ProjectUpdate(SQLModel):
    name: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=500)
    is_active: bool | None = Field(default=None)


# Properties to receive via API when patching project settings.
class ProjectSettingsUpdate(SQLModel):
    tracing: bool | None = Field(
        default=None,
        description="Enable/disable Langfuse tracing for this project.",
    )


# Database model for Project
class Project(ProjectBase, table=True):
    """Database model for projects."""

    __table_args__ = (
        UniqueConstraint("name", "organization_id", name="uq_project_name_org_id"),
    )

    id: int = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the project"},
    )
    storage_path: UUID = Field(
        default_factory=uuid4,
        nullable=False,
        unique=True,
        sa_column_kwargs={"comment": "Unique UUID used for cloud storage path"},
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(
            JSONB,
            nullable=False,
            comment=(
                "Project-level settings (JSONB). Keys: 'tracing' (bool) — "
                "Langfuse tracing opt-in, off by default to conserve Langfuse "
                "rate-limit/credit budget. Gates tracing for both the response "
                "path and evaluations (which fall back to cosine-only scoring)."
            ),
        ),
    )

    # Foreign keys
    organization_id: int = Field(
        foreign_key="organization.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the project was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the project was last updated"},
    )

    # Relationships
    creds: list["Credential"] = Relationship(
        back_populates="project", cascade_delete=True
    )
    assistants: list["Assistant"] = Relationship(
        back_populates="project", cascade_delete=True
    )
    organization: Optional["Organization"] = Relationship(back_populates="project")
    collections: list["Collection"] = Relationship(
        back_populates="project", cascade_delete=True
    )
    fine_tuning: list["FineTuning"] = Relationship(
        back_populates="project", cascade_delete=True
    )
    openai_conversations: list["OpenAIConversation"] = Relationship(
        back_populates="project", cascade_delete=True
    )


# Properties to return via API
class ProjectPublic(ProjectBase):
    id: int
    organization_id: int
    settings: dict[str, JsonValue] = {}
    inserted_at: datetime
    updated_at: datetime


class ProjectsPublic(SQLModel):
    data: list[ProjectPublic]
    count: int
