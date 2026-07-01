import logging

from fastapi import HTTPException
from sqlmodel import Session, select

from app.core.util import now
from app.crud.user_project import deactivate_users_without_projects
from app.models import Project, ProjectCreate, UserProject

logger = logging.getLogger(__name__)


def create_project(*, session: Session, project_create: ProjectCreate) -> Project:
    project = get_project_by_name(
        session=session,
        organization_id=project_create.organization_id,
        project_name=project_create.name,
    )
    if project:
        logger.warning(
            f"[create_project] Project name already exists | 'project_id': {project.id}, "
            f"'name': {project.name}, 'is_active': {project.is_active}"
        )
        if project.is_active:
            raise HTTPException(
                409, "A project with this name already exists in this organization"
            )
        raise HTTPException(
            409,
            "A project with this name already exists in this organization but is inactive. Reactivate it instead of creating a new one.",
        )

    db_project = Project.model_validate(project_create)
    db_project.inserted_at = now()
    db_project.updated_at = now()
    session.add(db_project)
    session.commit()
    session.refresh(db_project)
    logger.info(
        f"[create_project] Project Created Successfully | 'project_id': {db_project.id}, 'name': {db_project.name}"
    )
    return db_project


def get_project_by_id(*, session: Session, project_id: int) -> Project | None:
    return session.get(Project, project_id)


def get_project_by_name(
    *, session: Session, project_name: str, organization_id: int
) -> Project | None:
    statement = select(Project).where(
        Project.name == project_name, Project.organization_id == organization_id
    )
    return session.exec(statement).first()


def get_projects_by_organization(
    *,
    session: Session,
    org_id: int,
    is_active: bool | None = True,
    search: str | None = None,
) -> list[Project]:
    """
    Return projects for an organization.

    By default only active projects are returned. Pass ``is_active=False`` to
    list soft-deleted projects (e.g. to selectively reactivate them), or
    ``is_active=None`` to return every project regardless of status. Pass
    ``search`` for a case-insensitive substring match on the project name.
    """
    statement = select(Project).where(Project.organization_id == org_id)
    if is_active is not None:
        statement = statement.where(Project.is_active.is_(is_active))
    if search and search.strip():
        statement = statement.where(Project.name.ilike(f"%{search.strip()}%"))
    return session.exec(statement).all()


def soft_delete_project(*, session: Session, project: Project) -> list[int]:
    """
    Soft-delete a project by marking it inactive, so it stops appearing in
    listings and can no longer be used. No rows are removed.

    User accounts are never deleted. Any user left without an accessible project
    afterwards is marked inactive so they can no longer log in. Returns the list
    of deactivated user IDs.
    """
    project_id = project.id

    affected_user_ids = session.exec(
        select(UserProject.user_id).where(UserProject.project_id == project_id)
    ).all()

    # Soft-delete the project.
    project.is_active = False
    session.add(project)
    session.flush()

    deactivated = deactivate_users_without_projects(
        session=session, user_ids=affected_user_ids
    )
    session.commit()
    logger.info(
        f"[soft_delete_project] Project soft-deleted | 'project_id': {project_id}, "
        f"deactivated_users: {len(deactivated)}"
    )
    return deactivated


def hard_delete_project(*, session: Session, project: Project) -> list[int]:
    """
    Permanently delete a project and everything owned by it — collections,
    documents, credentials, assistants, fine-tunings, conversations, and
    user-project mappings all cascade away. This cannot be undone.

    User *accounts* are still never deleted (a user may belong to other
    projects): any user left without an accessible project after the cascade is
    marked inactive instead. Returns the list of deactivated user IDs.
    """
    project_id = project.id

    # Capture affected users before the cascade removes their mappings.
    affected_user_ids = session.exec(
        select(UserProject.user_id).where(UserProject.project_id == project_id)
    ).all()

    session.delete(project)
    session.flush()

    deactivated = deactivate_users_without_projects(
        session=session, user_ids=affected_user_ids
    )
    session.commit()
    logger.info(
        f"[hard_delete_project] Project permanently deleted | "
        f"'project_id': {project_id}, deactivated_users: {len(deactivated)}"
    )
    return deactivated


def validate_project(session: Session, project_id: int) -> Project:
    """
    Ensures that an project exists and is active.
    """
    project = get_project_by_id(session=session, project_id=project_id)
    if not project:
        logger.warning(
            f"[validate_project] Project not found | 'project_id': {project_id}"
        )
        raise HTTPException(404, "Project not found")

    if not project.is_active:
        logger.warning(
            f"[validate_project] Project is not active | 'project_id': {project_id}"
        )
        raise HTTPException(404, "Project is not active")

    return project


def validate_project_belongs_to_organization(
    session: Session,
    project_id: int,
    organization_id: int,
) -> Project:
    """Ensure project exists/is active and belongs to the given organization."""
    project = validate_project(session=session, project_id=project_id)
    if project.organization_id != organization_id:
        logger.error(
            f"[validate_project_belongs_to_organization] Project-org mismatch | "
            f"'project_id': {project_id}, 'organization_id': {organization_id}, "
            f"'project_organization_id': {project.organization_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="project_id does not belong to organization_id",
        )
    return project
