import logging

from fastapi import HTTPException
from sqlmodel import Session, select

from app.core.util import now
from app.crud.user_project import deactivate_users_without_projects
from app.models import Organization, OrganizationCreate, Project, UserProject

logger = logging.getLogger(__name__)


def create_organization(
    *, session: Session, org_create: OrganizationCreate
) -> Organization:
    existing = get_organization_by_name(session=session, name=org_create.name)
    if existing:
        logger.warning(
            f"[create_organization] Organization name already exists | "
            f"'org_id': {existing.id}, 'name': {existing.name}, 'is_active': {existing.is_active}"
        )
        if existing.is_active:
            raise HTTPException(
                status_code=409,
                detail="An organization with this name already exists",
            )
        raise HTTPException(
            status_code=409,
            detail="An organization with this name already exists but is inactive. Reactivate it instead of creating a new one.",
        )

    db_org = Organization.model_validate(org_create)
    db_org.inserted_at = now()
    db_org.updated_at = now()
    session.add(db_org)
    session.commit()
    session.refresh(db_org)
    logger.info(
        f"[create_organization] Organization Created Successfully | 'org_id': {db_org.id}, 'name': {db_org.name}"
    )
    return db_org


# Get organization by ID
def get_organization_by_id(session: Session, org_id: int) -> Organization | None:
    statement = select(Organization).where(Organization.id == org_id)
    return session.exec(statement).first()


def get_organization_by_name(*, session: Session, name: str) -> Organization | None:
    statement = select(Organization).where(Organization.name == name)
    return session.exec(statement).first()


def cascade_deactivate_organization(
    *, session: Session, organization: Organization
) -> tuple[int, list[int]]:
    """
    Mark an organization inactive and cascade the deactivation to all of its
    projects, then deactivate any user left without an accessible project.

    Does not commit — the caller is responsible for committing. Returns a tuple
    of (number of projects deactivated, list of deactivated user IDs).
    """
    org_id = organization.id

    affected_user_ids = session.exec(
        select(UserProject.user_id).where(UserProject.organization_id == org_id)
    ).all()

    organization.is_active = False
    session.add(organization)

    projects = session.exec(
        select(Project).where(
            Project.organization_id == org_id,
            Project.is_active.is_(True),
        )
    ).all()
    for project in projects:
        project.is_active = False
        session.add(project)

    session.flush()

    deactivated = deactivate_users_without_projects(
        session=session, user_ids=affected_user_ids
    )
    return len(projects), deactivated


def soft_delete_organization(
    *, session: Session, organization: Organization
) -> list[int]:
    """
    Soft-delete an organization by marking it inactive. All of its projects are
    deactivated as well, so the org and everything under it stop appearing in
    listings and can no longer be used. No rows are removed.

    User accounts are never deleted. Any user left without an accessible project
    afterwards is marked inactive so they can no longer log in. Returns the list
    of deactivated user IDs.
    """
    org_id = organization.id
    project_count, deactivated = cascade_deactivate_organization(
        session=session, organization=organization
    )
    session.commit()
    logger.info(
        f"[soft_delete_organization] Organization soft-deleted | 'org_id': {org_id}, "
        f"deactivated_projects: {project_count}, deactivated_users: {len(deactivated)}"
    )
    return deactivated


def hard_delete_organization(
    *, session: Session, organization: Organization
) -> list[int]:
    """
    Permanently delete an organization and everything owned by it — projects,
    collections, documents, credentials, assistants, fine-tunings, conversations,
    and user-project mappings all cascade away. This cannot be undone.

    User *accounts* are still never deleted (a user may belong to other orgs):
    any user left without an accessible project after the cascade is marked
    inactive instead. Returns the list of deactivated user IDs.
    """
    org_id = organization.id

    # Capture affected users before the cascade removes their mappings.
    affected_user_ids = session.exec(
        select(UserProject.user_id).where(UserProject.organization_id == org_id)
    ).all()

    session.delete(organization)
    session.flush()

    deactivated = deactivate_users_without_projects(
        session=session, user_ids=affected_user_ids
    )
    session.commit()
    logger.info(
        f"[hard_delete_organization] Organization permanently deleted | "
        f"'org_id': {org_id}, deactivated_users: {len(deactivated)}"
    )
    return deactivated


# Validate if organization exists and is active
def validate_organization(session: Session, org_id: int) -> Organization:
    """
    Ensures that an organization exists and is active.
    """
    organization = get_organization_by_id(session, org_id)
    if not organization:
        logger.warning(
            f"[validate_organization] Organization not found | 'org_id': {org_id}"
        )
        raise HTTPException(404, "Organization not found")

    if not organization.is_active:
        logger.warning(
            f"[validate_organization] Organization is not active | 'org_id': {org_id}"
        )
        raise HTTPException(status_code=403, detail="Organization is not active")

    return organization
