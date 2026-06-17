import logging
import secrets
from collections.abc import Iterable, Sequence

from sqlmodel import Session, and_, select

from app.core.security import get_password_hash
from app.crud.auth import get_user_accessible_projects
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

    if not user:
        user = User(
            email=email,
            full_name=full_name,
            is_active=False,
            hashed_password=get_password_hash(secrets.token_urlsafe(16)),
        )
        session.add(user)
        session.flush()
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

    return user, "added"


def deactivate_users_without_projects(
    *, session: Session, user_ids: Iterable[int]
) -> list[int]:
    """
    Mark users inactive when they no longer belong to any *active* project.

    A user counts as having access only through a mapping to an active project
    in an active organization (see ``get_user_accessible_projects``), so this
    correctly handles soft-deleted orgs/projects whose mappings still exist.

    Users are never deleted: superusers are left untouched, and any user that
    still has at least one accessible project keeps their active status. A user
    left without any accessible project can no longer log in until they are
    added to one again. Returns the list of user IDs that were deactivated.
    """
    deactivated: list[int] = []
    for user_id in set(user_ids):
        if get_user_accessible_projects(session=session, user_id=user_id):
            continue

        user = session.get(User, user_id)
        if user and not user.is_superuser and user.is_active:
            user.is_active = False
            session.add(user)
            deactivated.append(user_id)

    if deactivated:
        session.flush()
        logger.info(
            f"[deactivate_users_without_projects] Users deactivated | user_ids: {deactivated}"
        )
    return deactivated


def reactivate_users_with_access(
    *, session: Session, user_ids: Iterable[int]
) -> list[int]:
    """
    Re-activate users who regained access to an active project.

    The inverse of ``deactivate_users_without_projects``: when an org or project
    is reactivated, users that had been deactivated (because they had lost all
    access) are made active again so they can log in. Their project mappings are
    never removed by a soft delete, so access is restored automatically once the
    org/project is active again. Superusers and already-active users are left
    untouched. Returns the list of user IDs that were reactivated.
    """
    reactivated: list[int] = []
    for user_id in set(user_ids):
        user = session.get(User, user_id)
        if user and not user.is_active and not user.is_superuser:
            if get_user_accessible_projects(session=session, user_id=user_id):
                user.is_active = True
                session.add(user)
                reactivated.append(user_id)

    if reactivated:
        session.flush()
        logger.info(
            f"[reactivate_users_with_access] Users reactivated | user_ids: {reactivated}"
        )
    return reactivated


def get_user_ids_for_project(*, session: Session, project_id: int) -> list[int]:
    """Return the IDs of all users mapped to a project."""
    return list(
        session.exec(
            select(UserProject.user_id).where(UserProject.project_id == project_id)
        ).all()
    )


def remove_user_from_project(
    *, session: Session, user_id: int, project_id: int
) -> bool:
    """
    Remove a user from a project. If this was their last project,
    deactivate the user account (the user is never deleted).

    Returns True if removed, False if not found.
    """
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

    deactivate_users_without_projects(session=session, user_ids=[user_id])

    return True


def get_user_projects(*, session: Session, user_id: int) -> Sequence[UserProject]:
    """Get all project mappings for a user."""
    statement = select(UserProject).where(UserProject.user_id == user_id)
    return session.exec(statement).all()
