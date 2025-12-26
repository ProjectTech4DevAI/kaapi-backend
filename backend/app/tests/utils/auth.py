from uuid import UUID

from sqlmodel import Session, select, SQLModel

from app.models import User, Organization, Project, APIKey
from app.core.config import settings


class TestAuthContext(SQLModel):
    """Authentication context for testing"""

    user_id: int
    project_id: int
    organization_id: int
    key: str  # The full unencrypted API key with "ApiKey " prefix
    api_key_id: UUID  # The UUID of the API key record

    user: User
    project: Project
    organization: Organization
    api_key: APIKey


def get_test_auth_context(
    session: Session,
    user_email: str,
    project_name: str,
    raw_key: str,
    user_type: str = "User",
) -> TestAuthContext:
    """
    Helper function to get authentication context from seeded data.

    Args:
        session: Database session
        user_email: Email of the user
        project_name: Name of the project
        raw_key: The full unencrypted API key with "ApiKey " prefix
        user_type: Type of user for error messages (e.g., "Superuser", "User")

    Returns:
        TestAuthContext with all IDs and keys from seeded data

    Raises:
        ValueError: If the required data is not found in the database
    """
    user = session.exec(select(User).where(User.email == user_email)).first()
    if not user:
        raise ValueError(
            f"{user_type} with email {user_email} not found. Ensure seed data is loaded."
        )

    project = session.exec(select(Project).where(Project.name == project_name)).first()
    if not project:
        raise ValueError(
            f"Project {project_name} not found. Ensure seed data is loaded."
        )

    org = session.exec(
        select(Organization).where(Organization.id == project.organization_id)
    ).first()
    if not org:
        raise ValueError(f"Organization for project {project_name} not found.")

    api_key = session.exec(
        select(APIKey)
        .where(APIKey.user_id == user.id)
        .where(APIKey.project_id == project.id)
        .where(APIKey.is_deleted == False)
    ).first()
    if not api_key:
        raise ValueError(
            f"API key for {user_type.lower()} and project {project_name} not found."
        )

    return TestAuthContext(
        user_id=user.id,
        project_id=project.id,
        organization_id=org.id,
        key=raw_key,
        api_key_id=api_key.id,
        user=user,
        project=project,
        organization=org,
        api_key=api_key,
    )


def get_superuser_test_auth_context(session: Session) -> TestAuthContext:
    """
    Get authentication context for superuser from seeded data.

    Uses SUPERUSER_EMAIL with Glific project based on seed_data.json:
    - User: {{SUPERUSER_EMAIL}} (is_superuser: true)
    - Project: Glific
    - API Key: ApiKey No3x47A5qoIGhm0kVKjQ77dhCqEdWRIQZlEPzzzh7i8

    Args:
        session: Database session

    Returns:
        TestAuthContext with all IDs and keys from seeded data

    Raises:
        ValueError: If the required data is not found in the database
    """
    return get_test_auth_context(
        session=session,
        user_email=settings.FIRST_SUPERUSER,
        project_name="Glific",
        raw_key="ApiKey No3x47A5qoIGhm0kVKjQ77dhCqEdWRIQZlEPzzzh7i8",
        user_type="Superuser",
    )


def get_user_test_auth_context(session: Session) -> TestAuthContext:
    """
    Get authentication context for normal user from seeded data.

    Uses ADMIN_EMAIL with Dalgo project based on seed_data.json:
    - User: {{ADMIN_EMAIL}} (is_superuser: false)
    - Project: Dalgo
    - API Key: ApiKey Px8y47B6roJHin1lWLkR88eiDrFdXSJRZmFQazzai8j9

    Args:
        session: Database session

    Returns:
        TestAuthContext with all IDs and keys from seeded data

    Raises:
        ValueError: If the required data is not found in the database
    """
    return get_test_auth_context(
        session=session,
        user_email=settings.EMAIL_TEST_USER,
        project_name="Dalgo",
        raw_key="ApiKey Px8y47B6roJHin1lWLkR88eiDrFdXSJRZmFQazzai8j",
        user_type="User",
    )
