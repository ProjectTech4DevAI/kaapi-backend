import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.crud.organization import create_organization, get_organization_by_id, validate_organization
from app.models import Organization, OrganizationCreate
from app.tests.utils.utils import random_lower_string, get_non_existent_id
from app.tests.utils.test_data import create_test_organization


def test_create_organization(db: Session) -> None:
    """Test creating an organization."""
    name = random_lower_string()
    org_in = OrganizationCreate(name=name)
    organization = create_organization(session=db, org_create=org_in)

    assert organization.name == name
    assert organization.id is not None
    assert organization.is_active is True


def test_get_organization_by_id(db: Session) -> None:
    """Test retrieving an organization by ID."""
    organization = create_test_organization(db)

    fetched_org = get_organization_by_id(session=db, org_id=organization.id)
    assert fetched_org
    assert fetched_org.id == organization.id
    assert fetched_org.name == organization.name


def test_get_non_existent_organization(db: Session) -> None:
    """Test retrieving a non-existent organization should return None."""
    organization_id = get_non_existent_id(db, Organization)
    fetched_org = get_organization_by_id(session=db, org_id=organization_id)
    assert fetched_org is None


def test_validate_organization_success(db: Session) -> None:
    """Test that a valid and active organization passes validation."""
    organization = create_test_organization(db)

    validated_org = validate_organization(session=db, org_id=organization.id)
    assert validated_org.id == organization.id


def test_validate_organization_not_found(db: Session) -> None:
    """Test that validation fails when organization does not exist."""
    non_existent_org_id = get_non_existent_id(db, Organization)
    with pytest.raises(HTTPException, match="Organization not found"):
        validate_organization(session=db, org_id=non_existent_org_id)


def test_validate_organization_inactive(db: Session) -> None:
    """Test that validation fails when organization is inactive."""
    organization = create_test_organization(db)
    organization.is_active = False
    db.add(organization)
    db.commit()
    db.refresh(organization)

    with pytest.raises(HTTPException, match="Organization is not active"):
        validate_organization(session=db, org_id=organization.id)
