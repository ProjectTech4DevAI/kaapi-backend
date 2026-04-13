from datetime import datetime

from pydantic import EmailStr
from sqlmodel import Field, SQLModel, UniqueConstraint

from app.core.util import now


class UserProjectBase(SQLModel):
    """Base model for user-project mapping."""

    user_id: int = Field(
        foreign_key="user.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the user"},
    )
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


class UserProject(UserProjectBase, table=True):
    """Maps users to projects within organizations."""

    __tablename__ = "user_project"
    __table_args__ = (
        UniqueConstraint("user_id", "project_id", name="uq_user_project"),
    )

    id: int = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the user-project mapping"},
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the mapping was created"},
    )


class UserEntry(SQLModel):
    """A single user entry with email and optional name."""

    email: EmailStr
    full_name: str | None = Field(default=None, max_length=255)


class AddUsersToProjectRequest(SQLModel):
    """Request to add one or more users to a project."""

    organization_id: int
    project_id: int
    users: list[UserEntry] = Field(min_length=1)


class UserProjectPublic(SQLModel):
    """Public response model for a user in a project."""

    user_id: int
    email: str
    full_name: str | None
    is_active: bool
    inserted_at: datetime
