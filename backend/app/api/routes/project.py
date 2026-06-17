import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlmodel import select

from app.api.deps import SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.organization import get_organization_by_id, validate_organization
from app.crud.project import (
    create_project,
    get_project_by_id,
    get_project_by_name,
    get_projects_by_organization,
    hard_delete_project,
    soft_delete_project,
)
from app.crud.user_project import (
    deactivate_users_without_projects,
    get_user_ids_for_project,
    reactivate_users_with_access,
)
from app.models import (
    DeleteRequest,
    Project,
    ProjectCreate,
    ProjectPublic,
    ProjectUpdate,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects", tags=["Projects"])


# Retrieve projects
@router.get(
    "",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[list[ProjectPublic]],
    description=load_description("projects/list.md"),
)
def read_projects(
    session: SessionDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    search: str
    | None = Query(
        None, description="Case-insensitive substring match on the project name"
    ),
    is_active: bool = Query(
        True,
        description="Filter by active status. Pass false to list soft-deleted projects.",
    ),
):
    filters = [Project.is_active.is_(is_active)]
    if search and search.strip():
        filters.append(Project.name.ilike(f"%{search.strip()}%"))

    count_statement = select(func.count()).select_from(Project).where(*filters)
    count = session.exec(count_statement).one()

    statement = select(Project).where(*filters).offset(skip).limit(limit)
    projects = session.exec(statement).all()

    has_more = (skip + limit) < count
    return APIResponse.success_response(projects, metadata={"has_more": has_more})


# Create a new project
@router.post(
    "",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[ProjectPublic],
    description=load_description("projects/create.md"),
)
def create_new_project(*, session: SessionDep, project_in: ProjectCreate):
    project = create_project(session=session, project_create=project_in)
    return APIResponse.success_response(project)


@router.get(
    "/{project_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[ProjectPublic],
    description=load_description("projects/get.md"),
)
def read_project(*, session: SessionDep, project_id: int):
    """
    Retrieve a project by ID.
    """
    project = get_project_by_id(session=session, project_id=project_id)
    if project is None or not project.is_active:
        raise HTTPException(status_code=404, detail="Project not found")
    return APIResponse.success_response(project)


# Update a project
@router.patch(
    "/{project_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[ProjectPublic],
    description=load_description("projects/update.md"),
)
def update_project(*, session: SessionDep, project_id: int, project_in: ProjectUpdate):
    project = get_project_by_id(session=session, project_id=project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    project_data = project_in.model_dump(exclude_unset=True)

    # Reject renaming to a name already used by another project in the same org
    # (active or not), matching the uniqueness rule enforced on create.
    new_name = project_data.get("name")
    if new_name and new_name != project.name:
        existing = get_project_by_name(
            session=session,
            project_name=new_name,
            organization_id=project.organization_id,
        )
        if existing and existing.id != project.id:
            raise HTTPException(
                status_code=409,
                detail="A project with this name already exists in this organization",
            )

    # Detect an is_active transition so we can keep users in sync.
    target_active = project_data.get("is_active")
    activating = target_active is True and not project.is_active
    deactivating = target_active is False and project.is_active

    # A project cannot be active while its organization is inactive — reactivate
    # the organization first.
    if activating:
        org = get_organization_by_id(session=session, org_id=project.organization_id)
        if org is None or not org.is_active:
            raise HTTPException(
                status_code=400,
                detail="Cannot activate a project whose organization is inactive. Reactivate the organization first.",
            )

    project.sqlmodel_update(project_data)
    session.add(project)
    session.flush()

    if activating:
        # The project is live again — restore its previously deactivated users.
        reactivate_users_with_access(
            session=session,
            user_ids=get_user_ids_for_project(session=session, project_id=project.id),
        )
    elif deactivating:
        # Mirror the soft-delete: deactivate users left without an active project.
        deactivate_users_without_projects(
            session=session,
            user_ids=get_user_ids_for_project(session=session, project_id=project.id),
        )

    session.commit()
    session.refresh(project)
    logger.info(
        f"[update_project] Project updated successfully | project_id={project.id}"
    )
    return APIResponse.success_response(project)


# Delete a project
@router.delete(
    "/{project_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[None],
    description=load_description("projects/delete.md"),
)
def delete_project_endpoint(
    session: SessionDep,
    project_id: int,
    body: DeleteRequest | None = None,
):
    project = get_project_by_id(session=session, project_id=project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    hard_delete = body.hard_delete if body else False
    if hard_delete:
        hard_delete_project(session=session, project=project)
    else:
        soft_delete_project(session=session, project=project)

    logger.info(
        f"[delete_project_endpoint] Project deleted | project_id={project_id}, "
        f"hard_delete: {hard_delete}"
    )
    return APIResponse.success_response(None)


# Get projects by organization
@router.get(
    "/organization/{org_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[list[ProjectPublic]],
    description=load_description("projects/list_by_org.md"),
)
def read_projects_by_organization(
    session: SessionDep,
    org_id: int,
    is_active: bool = Query(
        True,
        description="Filter by active status. Pass false to list soft-deleted projects to reactivate.",
    ),
) -> APIResponse[list[ProjectPublic]]:
    validate_organization(session=session, org_id=org_id)
    projects = get_projects_by_organization(
        session=session, org_id=org_id, is_active=is_active
    )
    return APIResponse.success_response(projects)
