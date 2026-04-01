import logging
import secrets
from typing import Sequence

from sqlmodel import Session, and_, select

from app.core.security import get_password_hash
from app.models import (
    User,
    UserProject,
    UserProjectPublic,
)

logger = logging.getLogger(__name__)


def get_users_by_project(
    *, session: Session, project_id: int
) -> list[UserProjectPublic]:
    """Get all users mapped to a project."""
    statement = (
        select(
            User.id, User.email, User.full_name, User.is_active, UserProject.inserted_at
        )
        .join(UserProject, UserProject.user_id == User.id)
        .where(UserProject.project_id == project_id)
        .order_by(UserProject.inserted_at.desc())
    )
    results = session.exec(statement).all()
    return [
        UserProjectPublic(
            user_id=user_id,
            email=email,
            full_name=full_name,
            is_active=is_active,
            inserted_at=inserted_at,
        )
        for user_id, email, full_name, is_active, inserted_at in results
    ]


def add_user_to_project(
    *,
    session: Session,
    email: str,
    organization_id: int,
    project_id: int,
    full_name: str | None = None,
) -> tuple[User, str]:
    """
    Add a user to a project. Creates the user if they don't exist (is_active=False).

    Returns:
        Tuple of (user, status) where status is one of:
        - "added": User was successfully added to the project
        - "same_project": User is already in this project
        - "different_project": User is already assigned to another project
    """
    user = session.exec(select(User).where(User.email == email)).first()
    created = False

    if not user:
        user = User(
            email=email,
            full_name=full_name,
            is_active=False,
            hashed_password=get_password_hash(secrets.token_urlsafe(16)),
        )
        session.add(user)
        session.flush()
        created = True
        logger.info(
            f"[add_user_to_project] New user created | email: {email}, user_id: {user.id}"
        )
    elif full_name and not user.full_name:
        user.full_name = full_name
        session.add(user)
        session.flush()

    # Check if user is already assigned to any project
    existing = session.exec(
        select(UserProject).where(UserProject.user_id == user.id)
    ).first()

    if existing:
        if existing.project_id == project_id:
            return user, "same_project"
        else:
            return user, "different_project"

    user_project = UserProject(
        user_id=user.id,
        organization_id=organization_id,
        project_id=project_id,
    )
    session.add(user_project)
    session.flush()

    logger.info(
        f"[add_user_to_project] User added to project | user_id: {user.id}, project_id: {project_id}"
    )
    return user, "added"


def remove_user_from_project(
    *, session: Session, user_id: int, project_id: int
) -> bool:
    """Remove a user from a project. Returns True if removed, False if not found."""
    user_project = session.exec(
        select(UserProject).where(
            and_(
                UserProject.user_id == user_id,
                UserProject.project_id == project_id,
            )
        )
    ).first()

    if not user_project:
        return False

    session.delete(user_project)
    session.flush()

    logger.info(
        f"[remove_user_from_project] User removed from project | user_id: {user_id}, project_id: {project_id}"
    )
    return True


def get_user_projects(*, session: Session, user_id: int) -> Sequence[UserProject]:
    """Get all project mappings for a user."""
    statement = select(UserProject).where(UserProject.user_id == user_id)
    return session.exec(statement).all()
