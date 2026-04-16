import logging

from sqlmodel import Session, and_, select

from app.models import (
    APIKey,
    Organization,
    Project,
    UserProject,
)

logger = logging.getLogger(__name__)


def get_user_accessible_projects(*, session: Session, user_id: int) -> list[dict]:
    """
    Query distinct org/project pairs for a user from both
    the UserProject table and the APIKey table (backward compatibility).
    """
    # Query from UserProject table
    from_user_project = (
        select(Organization.id, Organization.name, Project.id, Project.name)
        .select_from(UserProject)
        .join(Organization, Organization.id == UserProject.organization_id)
        .join(Project, Project.id == UserProject.project_id)
        .where(
            and_(
                UserProject.user_id == user_id,
                Organization.is_active.is_(True),
                Project.is_active.is_(True),
            )
        )
    )

    # Query from APIKey table (backward compatibility)
    from_api_key = (
        select(Organization.id, Organization.name, Project.id, Project.name)
        .select_from(APIKey)
        .join(Organization, Organization.id == APIKey.organization_id)
        .join(Project, Project.id == APIKey.project_id)
        .where(
            and_(
                APIKey.user_id == user_id,
                APIKey.is_deleted.is_(False),
                Organization.is_active.is_(True),
                Project.is_active.is_(True),
            )
        )
    )

    # Union both queries and deduplicate
    combined = from_user_project.union(from_api_key)
    results = session.exec(combined).all()

    return [
        {
            "organization_id": org_id,
            "organization_name": org_name,
            "project_id": proj_id,
            "project_name": proj_name,
        }
        for org_id, org_name, proj_id, proj_name in results
    ]
