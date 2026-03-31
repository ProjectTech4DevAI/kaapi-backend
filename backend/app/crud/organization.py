import logging
from typing import Any, Optional
from datetime import datetime, timezone
from sqlmodel import Session, select
from fastapi import HTTPException

from app.models import Organization, OrganizationCreate
from app.core.util import now

logger = logging.getLogger(__name__)


def create_organization(
    *, session: Session, org_create: OrganizationCreate
) -> Organization:
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
def get_organization_by_id(session: Session, org_id: int) -> Optional[Organization]:
    statement = select(Organization).where(Organization.id == org_id)
    return session.exec(statement).first()


def get_organization_by_name(*, session: Session, name: str) -> Optional[Organization]:
    statement = select(Organization).where(Organization.name == name)
    return session.exec(statement).first()


# Validate if organization exists and is active
def validate_organization(session: Session, org_id: int) -> Organization:
    """
    Ensures that an organization exists and is active.
    """
    organization = get_organization_by_id(session, org_id)
    if not organization:
        logger.error(
            f"[validate_organization] Organization not found | 'org_id': {org_id}"
        )
        raise HTTPException(404, "Organization not found")

    if not organization.is_active:
        logger.error(
            f"[validate_organization] Organization is not active | 'org_id': {org_id}"
        )
        raise HTTPException(status_code=503, detail="Organization is not active")

    return organization
