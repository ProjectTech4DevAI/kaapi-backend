import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlmodel import select

from app.models import (
    Organization,
    OrganizationCreate,
    OrganizationUpdate,
    OrganizationPublic,
)
from app.api.deps import SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.organization import create_organization, get_organization_by_id
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/organizations", tags=["Organizations"])


# Retrieve organizations
@router.get(
    "/",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[List[OrganizationPublic]],
    description=load_description("organization/list.md"),
)
def read_organizations(session: SessionDep, skip: int = 0, limit: int = 100):
    count_statement = select(func.count()).select_from(Organization)
    count = session.exec(count_statement).one()

    statement = select(Organization).offset(skip).limit(limit)
    organizations = session.exec(statement).all()

    return APIResponse.success_response(organizations)


# Create a new organization
@router.post(
    "/",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[OrganizationPublic],
    description=load_description("organization/create.md"),
)
def create_new_organization(*, session: SessionDep, org_in: OrganizationCreate):
    new_org = create_organization(session=session, org_create=org_in)
    return APIResponse.success_response(new_org)


@router.get(
    "/{org_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[OrganizationPublic],
    description=load_description("organization/get.md"),
)
def read_organization(*, session: SessionDep, org_id: int):
    """
    Retrieve an organization by ID.
    """
    org = get_organization_by_id(session=session, org_id=org_id)
    if org is None:
        logger.error(f"[read_organization] Organization not found | org_id={org_id}")
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
):
    org = get_organization_by_id(session=session, org_id=org_id)
    if org is None:
        logger.error(
            f"[update_organization] Organization not found | 'org_id': {org_id}"
        )
        raise HTTPException(status_code=404, detail="Organization not found")

    org_data = org_in.model_dump(exclude_unset=True)
    org = org.model_copy(update=org_data)

    session.add(org)
    session.commit()
    session.flush()
    logger.info(
        f"[update_organization] Organization Updated Successfully | 'org_id': {org.id}"
    )
    return APIResponse.success_response(org)


# Delete an organization
@router.delete(
    "/{org_id}",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    response_model=APIResponse[None],
    include_in_schema=False,
    description=load_description("organization/delete.md"),
)
def delete_organization(session: SessionDep, org_id: int):
    org = get_organization_by_id(session=session, org_id=org_id)
    if org is None:
        logger.error(
            f"[delete_organization] Organization not found | 'org_id': {org_id}"
        )
        raise HTTPException(status_code=404, detail="Organization not found")

    session.delete(org)
    session.commit()
    logger.info(
        f"[delete_organization] Organization Deleted Successfully | 'org_id': {org_id}"
    )
    return APIResponse.success_response(None)
