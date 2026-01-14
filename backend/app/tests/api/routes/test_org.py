import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Organization
from app.main import app
from app.crud.organization import get_organization_by_id
from app.tests.utils.test_data import create_test_organization

client = TestClient(app)


@pytest.fixture
def test_organization(db: Session):
    return create_test_organization(db)


# Test creating an organization
def test_create_organization(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    org_name = "Test-Org"
    org_data = {"name": org_name, "is_active": True}
    response = client.post(
        f"{settings.API_V1_STR}/organizations/",
        json=org_data,
        headers=superuser_token_headers,
    )

    assert 200 <= response.status_code < 300
    created_org = response.json()
    assert "data" in created_org  # Make sure there's a 'data' field
    created_org_data = created_org["data"]
    org = get_organization_by_id(session=db, org_id=created_org_data["id"])
    assert org is not None  # The organization should be found in the DB
    assert org.name == created_org_data["name"]
    assert org.is_active == created_org_data["is_active"]


# Test retrieving organizations
def test_read_organizations(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/organizations/", headers=superuser_token_headers
    )
    assert response.status_code == 200
    response_data = response.json()
    assert "data" in response_data
    assert isinstance(response_data["data"], list)


# Updating an organization
def test_update_organization(
    db: Session,
    test_organization: Organization,
    superuser_token_headers: dict[str, str],
) -> None:
    updated_name = "UpdatedOrg"
    update_data = {"name": updated_name, "is_active": False}

    response = client.patch(
        f"{settings.API_V1_STR}/organizations/{test_organization.id}",
        json=update_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    updated_org = response.json()["data"]
    assert "name" in updated_org
    assert updated_org["name"] == update_data["name"]
    assert "is_active" in updated_org
    assert updated_org["is_active"] == update_data["is_active"]


# Test deleting an organization
def test_delete_organization(
    db: Session,
    test_organization: Organization,
    superuser_token_headers: dict[str, str],
) -> None:
    response = client.delete(
        f"{settings.API_V1_STR}/organizations/{test_organization.id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    response = client.get(
        f"{settings.API_V1_STR}/organizations/{test_organization.id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 404
