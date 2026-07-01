import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.crud.project import create_project, get_project_by_id
from app.main import app
from app.models import Organization, Project, ProjectCreate
from app.tests.utils.test_data import create_test_organization, create_test_project
from app.tests.utils.utils import random_lower_string

client = TestClient(app)


@pytest.fixture
def test_project(db: Session) -> Project:
    return create_test_project(db)


def _make_project(
    db: Session, org: Organization, name: str, is_active: bool
) -> Project:
    """Create a project with a given name and active status under an org."""
    return create_project(
        session=db,
        project_create=ProjectCreate(
            name=name,
            description="Test project",
            is_active=is_active,
            organization_id=org.id,
        ),
    )


# Test creating a project
def test_create_new_project(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    organization = create_test_organization(db)

    unique_project_name = "TestProject"
    project_description = "This is a test project description."
    project_data = ProjectCreate(
        name=unique_project_name,
        description=project_description,
        is_active=True,
        organization_id=organization.id,
    )

    response = client.post(
        f"{settings.API_V1_STR}/projects",
        json=project_data.dict(),
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    created_project = response.json()

    # Adjusted for a nested structure, if needed
    assert "data" in created_project  # Check if response contains a 'data' field
    assert (
        created_project["data"]["name"] == unique_project_name
    )  # Now checking 'name' inside 'data'
    assert created_project["data"]["description"] == project_description
    assert created_project["data"]["organization_id"] == organization.id


# Test retrieving projects
def test_read_projects(db: Session, superuser_token_headers: dict[str, str]) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/projects", headers=superuser_token_headers
    )
    assert response.status_code == 200
    response_data = response.json()
    assert "data" in response_data
    assert isinstance(response_data["data"], list)


# Test pagination has_more metadata for projects
def test_read_projects_has_more(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    create_test_project(db)
    create_test_project(db)

    response = client.get(
        f"{settings.API_V1_STR}/projects?skip=0&limit=1",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    response_data = response.json()
    assert "metadata" in response_data
    assert response_data["metadata"]["has_more"] is True

    response = client.get(
        f"{settings.API_V1_STR}/projects?skip=0&limit=100",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    response_data = response.json()
    assert "metadata" in response_data
    assert response_data["metadata"]["has_more"] is False


# Test the search param filters projects by name (defaults to active only)
def test_read_projects_search(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    org = create_test_organization(db)
    prefix = random_lower_string()
    _make_project(db, org, f"{prefix}-alpha", is_active=True)
    _make_project(db, org, f"{prefix}-beta", is_active=True)
    _make_project(db, org, "unrelated-name", is_active=True)

    response = client.get(
        f"{settings.API_V1_STR}/projects?search={prefix}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    names = [p["name"] for p in response.json()["data"]]
    assert f"{prefix}-alpha" in names
    assert f"{prefix}-beta" in names
    assert "unrelated-name" not in names


# Test the is_active param toggles active vs inactive projects, together with search
def test_read_projects_active_inactive_filter(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    org = create_test_organization(db)
    prefix = random_lower_string()
    _make_project(db, org, f"{prefix}-active", is_active=True)
    _make_project(db, org, f"{prefix}-inactive", is_active=False)

    # Default (is_active omitted) returns only active matches.
    response = client.get(
        f"{settings.API_V1_STR}/projects?search={prefix}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    names = [p["name"] for p in response.json()["data"]]
    assert f"{prefix}-active" in names
    assert f"{prefix}-inactive" not in names

    # is_active=false returns only inactive matches.
    response = client.get(
        f"{settings.API_V1_STR}/projects?search={prefix}&is_active=false",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    names = [p["name"] for p in response.json()["data"]]
    assert f"{prefix}-inactive" in names
    assert f"{prefix}-active" not in names


# Test updating a project
def test_update_project(
    db: Session, test_project: Project, superuser_token_headers: dict[str, str]
) -> None:
    update_data = {"name": "Updated Project Name", "is_active": False}

    response = client.patch(
        f"{settings.API_V1_STR}/projects/{test_project.id}",
        json=update_data,
        headers=superuser_token_headers,
    )

    assert response.status_code == 200
    updated_project = response.json()["data"]
    assert "name" in updated_project
    assert updated_project["name"] == update_data["name"]
    assert "is_active" in updated_project
    assert updated_project["is_active"] == update_data["is_active"]


# Test soft deleting a project (default)
def test_delete_project(
    db: Session, test_project: Project, superuser_token_headers: dict[str, str]
) -> None:
    project_id = test_project.id
    response = client.delete(
        f"{settings.API_V1_STR}/projects/{project_id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200

    response = client.get(
        f"{settings.API_V1_STR}/projects/{project_id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 404

    db.expire_all()
    project = get_project_by_id(session=db, project_id=project_id)
    assert project is not None
    assert project.is_active is False


def test_hard_delete_project(
    db: Session, test_project: Project, superuser_token_headers: dict[str, str]
) -> None:
    project_id = test_project.id
    response = client.request(
        "DELETE",
        f"{settings.API_V1_STR}/projects/{project_id}",
        json={"hard_delete": True},
        headers=superuser_token_headers,
    )
    assert response.status_code == 200

    # The row is permanently gone.
    db.expire_all()
    assert get_project_by_id(session=db, project_id=project_id) is None


# Test retrieving projects by organization
def test_read_projects_by_organization(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    project = create_test_project(db)
    response = client.get(
        f"{settings.API_V1_STR}/projects/organization/{project.organization_id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    response_data = response.json()
    assert "data" in response_data
    assert isinstance(response_data["data"], list)
    assert len(response_data["data"]) >= 1
    assert any(p["id"] == project.id for p in response_data["data"])


# Test the is_active param toggles active/inactive projects for an organization
def test_read_projects_by_organization_active_inactive(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    org = create_test_organization(db)
    active = _make_project(db, org, f"{random_lower_string()}-active", is_active=True)
    inactive = _make_project(
        db, org, f"{random_lower_string()}-inactive", is_active=False
    )

    # Default returns only active projects.
    response = client.get(
        f"{settings.API_V1_STR}/projects/organization/{org.id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    ids = [p["id"] for p in response.json()["data"]]
    assert active.id in ids
    assert inactive.id not in ids

    # is_active=false returns only inactive projects.
    response = client.get(
        f"{settings.API_V1_STR}/projects/organization/{org.id}?is_active=false",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    ids = [p["id"] for p in response.json()["data"]]
    assert inactive.id in ids
    assert active.id not in ids


# Test search combined with is_active for the org-scoped project list
def test_read_projects_by_organization_search(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    org = create_test_organization(db)
    prefix = random_lower_string()
    active = _make_project(db, org, f"{prefix}-active", is_active=True)
    inactive = _make_project(db, org, f"{prefix}-inactive", is_active=False)
    _make_project(db, org, "other-project", is_active=True)

    # Search active projects by name.
    response = client.get(
        f"{settings.API_V1_STR}/projects/organization/{org.id}?search={prefix}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    names = [p["name"] for p in response.json()["data"]]
    assert active.name in names
    assert "other-project" not in names
    assert inactive.name not in names

    # Search inactive projects by name.
    response = client.get(
        f"{settings.API_V1_STR}/projects/organization/{org.id}"
        f"?search={prefix}&is_active=false",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    names = [p["name"] for p in response.json()["data"]]
    assert inactive.name in names
    assert active.name not in names


# Test retrieving projects by non-existent organization
def test_read_projects_by_organization_not_found(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/projects/organization/999999",
        headers=superuser_token_headers,
    )
    assert response.status_code == 404


# Test retrieving projects by inactive organization
def test_read_projects_by_inactive_organization(
    db: Session, superuser_token_headers: dict[str, str]
) -> None:
    org = create_test_organization(db)
    org.is_active = False
    db.add(org)
    db.commit()
    db.refresh(org)

    response = client.get(
        f"{settings.API_V1_STR}/projects/organization/{org.id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 403
