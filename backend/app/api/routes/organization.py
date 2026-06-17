import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlmodel import select

from app.api.deps import SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.organization import (
    cascade_deactivate_organization,
    create_organization,
    get_organization_by_id,
    get_organization_by_name,
    hard_delete_organization,
    soft_delete_organization,
)
from app.models import (
    DeleteRequest,
    Organization,
    OrganizationCreate,
    OrganizationPublic,
    OrganizationUpdate,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/organizations", tags=["Organizations"])


# Retrieve organizations
@router.get(
    "",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[list[OrganizationPublic]],
    description=load_description("organization/list.md"),
)
def read_organizations(
    session: SessionDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    search: str
    | None = Query(
        None, description="Case-insensitive substring match on the organization name"
    ),
    is_active: bool = Query(
        True,
        description="Filter by active status. Pass false to list soft-deleted organizations.",
    ),
) -> APIResponse[list[OrganizationPublic]]:
    filters = [Organization.is_active.is_(is_active)]
    if search and search.strip():
        filters.append(Organization.name.ilike(f"%{search.strip()}%"))

    count_statement = select(func.count()).select_from(Organization).where(*filters)
    count = session.exec(count_statement).one()

    statement = select(Organization).where(*filters).offset(skip).limit(limit)
    organizations = session.exec(statement).all()

    has_more = (skip + limit) < count
    return APIResponse.success_response(organizations, metadata={"has_more": has_more})


# Create a new organization
@router.post(
    "",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[OrganizationPublic],
    description=load_description("organization/create.md"),
)
def create_new_organization(
    *, session: SessionDep, org_in: OrganizationCreate
) -> APIResponse[OrganizationPublic]:
    new_org = create_organization(session=session, org_create=org_in)
    return APIResponse.success_response(new_org)


@router.get(
    "/{org_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[OrganizationPublic],
    description=load_description("organization/get.md"),
)
def read_organization(
    *, session: SessionDep, org_id: int
) -> APIResponse[OrganizationPublic]:
    """
    Retrieve an organization by ID.
    """
    org = get_organization_by_id(session=session, org_id=org_id)
    if org is None or not org.is_active:
        raise HTTPException(status_code=404, detail="Organization not found")
    return APIResponse.success_response(org)


# Update an organization
@router.patch(
    "/{org_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[OrganizationPublic],
    description=load_description("organization/update.md"),
)
def update_organization(
    *, session: SessionDep, org_id: int, org_in: OrganizationUpdate
) -> APIResponse[OrganizationPublic]:
    org = get_organization_by_id(session=session, org_id=org_id)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    org_data = org_in.model_dump(exclude_unset=True)

    # Reject renaming to a name already taken by another organization (active or
    # not), matching the uniqueness rule enforced on create.
    new_name = org_data.get("name")
    if new_name and new_name != org.name:
        existing = get_organization_by_name(session=session, name=new_name)
        if existing and existing.id != org.id:
            raise HTTPException(
                status_code=409,
                detail="An organization with this name already exists",
            )

    # Detect an is_active transition so we can cascade consistently with delete.
    target_active = org_data.get("is_active")
    deactivating = target_active is False and org.is_active

    org.sqlmodel_update(org_data)
    session.add(org)
    session.flush()

    if deactivating:
        # Mirror the soft-delete: deactivate child projects and orphaned users.
        # Reactivating an org intentionally leaves its projects inactive so they
        # can be brought back selectively.
        cascade_deactivate_organization(session=session, organization=org)

    session.commit()
    session.refresh(org)
    logger.info(
        f"[update_organization] Organization Updated Successfully | 'org_id': {org.id}"
    )
    return APIResponse.success_response(org)


# Delete an organization
@router.delete(
    "/{org_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[None],
    description=load_description("organization/delete.md"),
)
def delete_organization_endpoint(
    session: SessionDep,
    org_id: int,
    body: DeleteRequest | None = None,
) -> APIResponse[None]:
    org = get_organization_by_id(session=session, org_id=org_id)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    hard_delete = body.hard_delete if body else False
    if hard_delete:
        hard_delete_organization(session=session, organization=org)
    else:
        soft_delete_organization(session=session, organization=org)

    logger.info(
        f"[delete_organization_endpoint] Organization deleted | 'org_id': {org_id}, "
        f"hard_delete: {hard_delete}"
    )
    return APIResponse.success_response(None)
