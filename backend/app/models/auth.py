from sqlmodel import Field, SQLModel
from app.models.user import User, UserPublic
from app.models.organization import Organization
from app.models.project import Project
from typing import TYPE_CHECKING


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None
    org_id: int | None = None
    project_id: int | None = None
    type: str = "access"


# Payload returned after verifying an invite JWT
class InviteTokenPayload(SQLModel):
    email: str
    organization_id: int
    project_id: int


# Google OAuth
class GoogleAuthRequest(SQLModel):
    token: str


class GoogleAuthResponse(SQLModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
    google_profile: dict
    requires_project_selection: bool = False
    available_projects: list[dict] = []


class SelectProjectRequest(SQLModel):
    project_id: int


class MagicLinkRequest(SQLModel):
    email: str


class AuthContext(SQLModel):
    user: User
    organization: Organization | None = None
    project: Project | None = None

    @property
    def organization_(self) -> Organization:
        """Non-optional organization - raises if None"""
        if self.organization is None:
            raise ValueError("Organization is required but was None")
        return self.organization

    @property
    def project_(self) -> Project:
        """Non-optional project - raises if None"""
        if self.project is None:
            raise ValueError("Project is required but was None")
        return self.project
