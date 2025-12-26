import random
import string
from uuid import UUID
from typing import Type, TypeVar

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.user import get_user_by_email
from app.models import Project, Assistant, Organization, Document


T = TypeVar("T")


def random_lower_string() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=32))


def generate_random_string(length: int = 10) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def random_email() -> str:
    return f"{random_lower_string()}@{random_lower_string()}.com"


def get_superuser_token_headers(client: TestClient) -> dict[str, str]:
    login_data = {
        "username": settings.FIRST_SUPERUSER,
        "password": settings.FIRST_SUPERUSER_PASSWORD,
    }
    r = client.post(f"{settings.API_V1_STR}/login/access-token", data=login_data)
    tokens = r.json()
    a_token = tokens["access_token"]
    headers = {"Authorization": f"Bearer {a_token}"}
    return headers


def get_user_id_by_email(db: Session) -> int:
    user = get_user_by_email(session=db, email=settings.EMAIL_TEST_USER)
    return user.id


def get_non_existent_id(session: Session, model: Type[T]) -> int:
    result = session.exec(select(model.id).order_by(model.id.desc())).first()
    return (result or 0) + 1


def get_project(session: Session, name: str | None = None) -> Project:
    """
    Retrieve an active project from the database.

    If a project name is provided, fetch the active project with that name.
    If no name is provided, fetch any random project.
    """
    if name:
        statement = (
            select(Project).where(Project.name == name, Project.is_active).limit(1)
        )
    else:
        statement = select(Project).where(Project.is_active).limit(1)

    project = session.exec(statement).first()

    if not project:
        raise ValueError("No active projects found")

    return project


def get_assistant(
    session: Session, project_id: int | None = None, name: str | None = None
) -> Assistant:
    """
    Retrieve an active assistant from the database.

    If a assistant name is provided, fetch the active assistant with that name.
    If no name is provided, fetch any random assistant.
    """
    filters = [Assistant.is_deleted == False]

    if project_id is not None:
        filters.append(Assistant.project_id == project_id)

    if name:
        filters.append(Assistant.name == name)

    statement = select(Assistant).where(*filters).limit(1)
    assistant = session.exec(statement).first()

    if not assistant:
        raise ValueError("No active assistants found")

    return assistant


def get_document(session: Session, fname: str | None = None) -> Document:
    """
    Retrieve an active document from the database.

    If a document filename is provided, fetch the active document with that name.
    If no name is provided, fetch any available document.
    """
    if fname:
        statement = (
            select(Document)
            .where(Document.fname == fname, Document.deleted_at.is_(None))
            .limit(1)
        )
    else:
        statement = select(Document).where(Document.deleted_at.is_(None)).limit(1)

    document = session.exec(statement).first()

    if not document:
        raise ValueError("No active documents found")

    return document


def get_organization(session: Session, name: str | None = None) -> Organization:
    """
    Retrieve an active organization from the database.

    If an organization name is provided, fetch the active organization with that name.
    If no name is provided, fetch any random organization.
    """
    if name:
        statement = (
            select(Organization)
            .where(Organization.name == name, Organization.is_active)
            .limit(1)
        )
    else:
        statement = select(Organization).where(Organization.is_active).limit(1)

    organization = session.exec(statement).first()

    if not organization:
        raise ValueError("No active organizations found")

    return organization


class SequentialUuidGenerator:
    def __init__(self, start: int = 0) -> None:
        self.start = start

    def __iter__(self) -> "SequentialUuidGenerator":
        return self

    def __next__(self) -> UUID:
        uu_id = UUID(int=self.start)
        self.start += 1
        return uu_id

    def peek(self) -> UUID:
        return UUID(int=self.start)
