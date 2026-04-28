import logging
from typing import List, Optional
from datetime import datetime, timezone
from sqlmodel import Session, select
from fastapi import HTTPException

from app.models import Project, ProjectCreate, Organization
from app.core.util import now

logger = logging.getLogger(__name__)


def create_project(*, session: Session, project_create: ProjectCreate) -> Project:
    project = get_project_by_name(
        session=session,
        organization_id=project_create.organization_id,
        project_name=project_create.name,
    )
    if project:
        logger.warning(
            f"[create_project] Project already exists | 'project_id': {project.id}, 'name': {project.name}"
        )
        raise HTTPException(409, "Project already exists")

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


def get_project_by_id(*, session: Session, project_id: int) -> Optional[Project]:
    return session.get(Project, project_id)


def get_project_by_name(
    *, session: Session, project_name: str, organization_id: int
) -> Optional[Project]:
    statement = select(Project).where(
        Project.name == project_name, Project.organization_id == organization_id
    )
    return session.exec(statement).first()


def get_projects_by_organization(*, session: Session, org_id: int) -> List[Project]:
    statement = select(Project).where(Project.organization_id == org_id)
    return session.exec(statement).all()


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
