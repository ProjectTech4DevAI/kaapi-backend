import json
import logging
from pathlib import Path
from typing import Optional, Any

from pydantic import BaseModel, EmailStr
from sqlmodel import Session, delete, select

from app.core.db import engine
from app.core import settings
from app.core.security import get_password_hash
from app.models import (
    APIKey,
    Organization,
    Project,
    User,
)


class OrgData(BaseModel):
    name: str
    is_active: bool


class ProjectData(BaseModel):
    name: str
    description: str
    is_active: bool
    organization_name: str


class UserData(BaseModel):
    email: EmailStr
    full_name: str
    is_superuser: bool
    is_active: bool
    password: str


class APIKeyData(BaseModel):
    organization_name: str
    project_name: str
    user_email: EmailStr
    api_key: str
    is_deleted: bool
    deleted_at: Optional[str] = None


def load_seed_data() -> dict[str, Any]:
    """Load seed data from JSON file."""
    json_path = Path(__file__).parent / "seed_data.json"
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"[seed_data] Seed data file not found at {json_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"[seed_data] Failed to decode JSON from {json_path}: {e}")
        raise


def create_organization(session: Session, org_data_raw: dict[str, Any]) -> Organization:
    """Create an organization from data."""
    try:
        org_data = OrgData.model_validate(org_data_raw)
        organization = Organization(name=org_data.name, is_active=org_data.is_active)
        session.add(organization)
        session.flush()  # Ensure ID is assigned
        return organization
    except Exception as e:
        logging.error(f"[seed_data] Error creating organization: {e}")
        raise


def create_project(session: Session, project_data_raw: dict[str, Any]) -> Project:
    """Create a project from data."""
    try:
        project_data = ProjectData.model_validate(project_data_raw)
        organization = session.exec(
            select(Organization).where(
                Organization.name == project_data.organization_name
            )
        ).first()
        if not organization:
            raise ValueError(
                f"Organization '{project_data.organization_name}' not found"
            )
        project = Project(
            name=project_data.name,
            description=project_data.description,
            is_active=project_data.is_active,
            organization_id=organization.id,
        )
        session.add(project)
        session.flush()  # Ensure ID is assigned
        return project
    except Exception as e:
        logging.error(f"[seed_data] Error creating project: {e}")
        raise


def create_user(session: Session, user_data_raw: dict[str, Any]) -> User:
    """Create a user from data."""
    try:
        user_data = UserData.model_validate(user_data_raw)
        hashed_password = get_password_hash(user_data.password)
        user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            is_superuser=user_data.is_superuser,
            is_active=user_data.is_active,
            hashed_password=hashed_password,
        )
        session.add(user)
        session.flush()  # Ensure ID is assigned
        return user
    except Exception as e:
        logging.error(f"[seed_data] Error creating user: {e}")
        raise


def create_api_key(session: Session, api_key_data_raw: dict[str, Any]) -> APIKey:
    """Create an API key from data."""
    try:
        api_key_data = APIKeyData.model_validate(api_key_data_raw)
        organization = session.exec(
            select(Organization).where(
                Organization.name == api_key_data.organization_name
            )
        ).first()
        if not organization:
            raise ValueError(
                f"Organization '{api_key_data.organization_name}' not found"
            )
        project = session.exec(
            select(Project).where(Project.name == api_key_data.project_name)
        ).first()
        if not project:
            raise ValueError(f"Project '{api_key_data.project_name}' not found")
        # Query user ID by email
        user = session.exec(
            select(User).where(User.email == api_key_data.user_email)
        ).first()
        if not user:
            raise ValueError(f"User '{api_key_data.user_email}' not found")

        # Extract key_prefix from the provided API key and hash the full key
        # API key format: "ApiKey {key_prefix}{random_key}" where key_prefix is 16 chars
        raw_key = api_key_data.api_key
        if not raw_key.startswith("ApiKey "):
            raise ValueError(f"Invalid API key format: {raw_key}")

        # Extract the key_prefix (first 16 characters after "ApiKey ")
        key_portion = raw_key[7:]  # Remove "ApiKey " prefix

        key_prefix = key_portion[:12]  # First 12 characters as prefix

        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        key_hash = pwd_context.hash(key_portion[12:])

        api_key = APIKey(
            organization_id=organization.id,
            project_id=project.id,
            user_id=user.id,
            key_prefix=key_prefix,
            key_hash=key_hash,
            is_deleted=api_key_data.is_deleted,
            deleted_at=api_key_data.deleted_at,
        )
        session.add(api_key)
        session.flush()
        return api_key
    except Exception as e:
        logging.error(f"[seed_data] Error creating API key: {e}")
        raise


def clear_database(session: Session) -> None:
    """Clear all seeded data from the database."""
    session.exec(delete(APIKey))
    session.exec(delete(Project))
    session.exec(delete(Organization))
    session.exec(delete(User))
    session.commit()
    logging.info("[seed_data] Existing database cleared")


def seed_database(session: Session) -> None:
    """
    Seed the database with initial test data.

    This function creates a complete test environment including:
    - Organizations (Project Tech4dev)
    - Projects (Glific, Dalgo)
    - Users (superuser, test user)
    - API Keys for both users

    This seed data is used by the test suite.
    """
    logging.info("[seed_data] Starting database seeding")
    try:
        clear_database(session)

        seed_data = load_seed_data()

        for org_data in seed_data["organization"]:
            create_organization(session, org_data)

        for user_data in seed_data["users"]:
            if user_data["email"] == "{{SUPERUSER_EMAIL}}":
                user_data["email"] = settings.FIRST_SUPERUSER
            elif user_data["email"] == "{{ADMIN_EMAIL}}":
                user_data["email"] = settings.EMAIL_TEST_USER

        for user_data in seed_data["users"]:
            create_user(session, user_data)

        for project_data in seed_data["projects"]:
            create_project(session, project_data)

        for api_key_data in seed_data["apikeys"]:
            if api_key_data["user_email"] == "{{SUPERUSER_EMAIL}}":
                api_key_data["user_email"] = settings.FIRST_SUPERUSER
            elif api_key_data["user_email"] == "{{ADMIN_EMAIL}}":
                api_key_data["user_email"] = settings.EMAIL_TEST_USER

        for api_key_data in seed_data["apikeys"]:
            create_api_key(session, api_key_data)

        session.commit()
        logging.info("[seed_data] Database seeded successfully")
    except Exception as e:
        logging.error(f"[seed_data] Error seeding database: {e}")
        session.rollback()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with Session(engine) as session:
        seed_database(session)
