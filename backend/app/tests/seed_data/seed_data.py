import json
import logging
from pathlib import Path
from passlib.context import CryptContext
from typing import Optional, Any

from pydantic import BaseModel, EmailStr
from sqlmodel import Session, delete, select

from app.core.db import engine
from app.core import settings
from app.core.security import get_password_hash, encrypt_credentials
from app.models import (
    APIKey,
    Organization,
    Project,
    User,
    Credential,
    Assistant,
    Document,
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


class CredentialData(BaseModel):
    is_active: bool
    provider: str
    credential: str
    organization_name: str
    project_name: str
    deleted_at: Optional[str] = None


class AssistantData(BaseModel):
    assistant_id: str
    name: str
    instructions: str
    model: str
    vector_store_ids: list[str]
    temperature: float
    max_num_results: int
    project_name: str
    organization_name: str


class DocumentData(BaseModel):
    fname: str
    object_store_url: str
    organization_name: str
    project_name: str


def load_seed_data() -> dict[str, Any]:
    """Load seed data from JSON file."""
    json_path = Path(__file__).parent / "seed_data.json"
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(
            f"[tests.seed_data]Error: Seed data file not found at {json_path}"
        )
        raise
    except json.JSONDecodeError as e:
        logging.error(
            f"[tests.seed_data]Error: Failed to decode JSON from {json_path}: {e}"
        )
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
        logging.error(f"[tests.seed_data]Error creating organization: {e}")
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
        logging.error(f"[tests.seed_data]Error creating project: {e}")
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
        session.flush()
        return user
    except Exception as e:
        logging.error(f"[tests.seed_data]Error creating user: {e}")
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
        key_portion = raw_key[7:]

        key_prefix = key_portion[:12]

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
        logging.error(f"[tests.seed_data]Error creating API key: {e}")
        raise


def create_credential(
    session: Session, credential_data_raw: dict[str, Any]
) -> Credential:
    """Create a credential from data."""
    try:
        credential_data = CredentialData.model_validate(credential_data_raw)
        organization = session.exec(
            select(Organization).where(
                Organization.name == credential_data.organization_name
            )
        ).first()
        if not organization:
            raise ValueError(
                f"Organization '{credential_data.organization_name}' not found"
            )

        project = session.exec(
            select(Project).where(Project.name == credential_data.project_name)
        ).first()
        if not project:
            raise ValueError(f"Project '{credential_data.project_name}' not found")

        # Encrypt the credential data - convert string to dict first, then encrypt
        credential_dict = json.loads(credential_data.credential)
        encrypted_credential = encrypt_credentials(credential_dict)

        credential = Credential(
            is_active=credential_data.is_active,
            provider=credential_data.provider,
            credential=encrypted_credential,
            organization_id=organization.id,
            project_id=project.id,
            deleted_at=credential_data.deleted_at,
        )
        session.add(credential)
        session.flush()  # Ensure ID is assigned
        return credential
    except Exception as e:
        logging.error(f"[tests.seed_data]Error creating credential: {e}")
        raise


def create_assistant(session: Session, assistant_data_raw: dict[str, Any]) -> Assistant:
    """Create an assistant from data."""
    try:
        assistant_data = AssistantData.model_validate(assistant_data_raw)
        organization = session.exec(
            select(Organization).where(
                Organization.name == assistant_data.organization_name
            )
        ).first()
        if not organization:
            raise ValueError(
                f"Organization '{assistant_data.organization_name}' not found"
            )

        project = session.exec(
            select(Project).where(Project.name == assistant_data.project_name)
        ).first()
        if not project:
            raise ValueError(f"Project '{assistant_data.project_name}' not found")

        assistant = Assistant(
            assistant_id=assistant_data.assistant_id,
            name=assistant_data.name,
            instructions=assistant_data.instructions,
            model=assistant_data.model,
            vector_store_ids=assistant_data.vector_store_ids,
            temperature=assistant_data.temperature,
            max_num_results=assistant_data.max_num_results,
            organization_id=organization.id,
            project_id=project.id,
        )
        session.add(assistant)
        session.flush()  # Ensure ID is assigned
        return assistant
    except Exception as e:
        logging.error(f"[tests.seed_data]Error creating assistant: {e}")
        raise


def create_document(session: Session, document_data_raw: dict[str, Any]) -> Document:
    """Create a document from seed data."""
    try:
        document_data = DocumentData.model_validate(document_data_raw)
        organization = session.exec(
            select(Organization).where(
                Organization.name == document_data.organization_name
            )
        ).first()

        if not organization:
            raise ValueError(
                f"Organization '{document_data.organization_name}' not found"
            )

        project = session.exec(
            select(Project).where(
                Project.name == document_data.project_name,
                Project.organization_id == organization.id,
            )
        ).first()
        if not project:
            raise ValueError(
                f"Project '{document_data.project_name}' not found in organization '{organization.name}'"
            )

        users = session.exec(
            select(User)
            .join(APIKey, APIKey.user_id == User.id)
            .where(APIKey.organization_id == organization.id)
        ).all()

        user = users[1]
        if not user:
            raise ValueError(f"No user found in organization '{organization.name}'")

        document = Document(
            fname=document_data.fname,
            object_store_url=document_data.object_store_url,
            project_id=project.id,
        )

        session.add(document)
        session.flush()
        return document

    except Exception as e:
        logging.error(f"[tests.seed_data]Error creating document: {e}")
        raise


def clear_database(session: Session) -> None:
    """Clear all seeded data from the database."""
    session.exec(delete(Assistant))
    session.exec(delete(Document))
    session.exec(delete(APIKey))
    session.exec(delete(Project))
    session.exec(delete(Organization))
    session.exec(delete(User))
    session.exec(delete(Credential))
    session.commit()
    logging.info("[tests.seed_data] Existing database cleared")


def seed_database(session: Session) -> None:
    """
    Seed the database with initial test data.

    This function creates a complete test environment including:
    - Organizations (Project Tech4dev)
    - Projects (Glific, Dalgo)
    - Users (superuser, test user)
    - API Keys for both users
    - OpenAI Credentials for both projects (ensures all tests have credentials)
    - Langfuse Credentials for both projects (for tracing and observability tests)
    - Test Assistants for both projects
    - Sample Documents

    This seed data is used by the test suite and ensures that all tests
    can rely on both OpenAI and Langfuse credentials being available without manual setup.
    """
    logging.info("[tests.seed_data] Starting database seeding")
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

        for credential_data in seed_data["credentials"]:
            create_credential(session, credential_data)

        for assistant_data in seed_data.get("assistants", []):
            create_assistant(session, assistant_data)

        for document_data in seed_data.get("documents", []):
            create_document(session, document_data)

        session.commit()
        logging.info("[tests.seed_data] Database seeded successfully")
    except Exception as e:
        logging.error(f"[tests.seed_data] Error seeding database: {e}")
        session.rollback()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with Session(engine) as session:
        seed_database(session)
