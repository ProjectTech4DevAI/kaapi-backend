import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.crud.project import (
    create_project,
    get_project_by_id,
    get_project_by_name,
    get_projects_by_organization,
    is_tracing_enabled,
    update_project_settings,
    validate_project,
)
from app.models import Project, ProjectCreate
from app.tests.utils.test_data import create_test_organization, create_test_project
from app.tests.utils.utils import get_non_existent_id, random_lower_string


def test_create_project(db: Session) -> None:
    """Test creating a project linked to an organization."""
    organization = create_test_organization(db)

    project_name = random_lower_string()
    project_data = ProjectCreate(
        name=project_name,
        description="Test description",
        is_active=True,
        organization_id=organization.id,
    )

    project = create_project(session=db, project_create=project_data)

    assert project.id is not None
    assert project.name == project_name
    assert project.description == "Test description"
    assert project.organization_id == organization.id


def test_create_project_duplicate_name(db: Session) -> None:
    """Test creating a project with a duplicate name."""
    organization = create_test_organization(db)

    project_name = random_lower_string()
    project_data = ProjectCreate(
        name=project_name,
        description="Test description",
        is_active=True,
        organization_id=organization.id,
    )
    create_project(session=db, project_create=project_data)
    with pytest.raises(HTTPException, match="already exists in this organization"):
        create_project(session=db, project_create=project_data)


def test_create_project_duplicate_inactive_name(db: Session) -> None:
    """Recreating a soft-deleted project name is rejected with a reactivate hint."""
    organization = create_test_organization(db)

    project_data = ProjectCreate(
        name=random_lower_string(),
        description="Test description",
        is_active=True,
        organization_id=organization.id,
    )
    project = create_project(session=db, project_create=project_data)

    project.is_active = False
    db.add(project)
    db.flush()

    with pytest.raises(HTTPException, match="inactive"):
        create_project(session=db, project_create=project_data)


def test_get_project_by_id(db: Session) -> None:
    """Test retrieving a project by ID."""
    project = create_test_project(db)

    fetched_project = get_project_by_id(session=db, project_id=project.id)
    assert fetched_project is not None
    assert fetched_project.id == project.id
    assert fetched_project.name == project.name


def test_get_project_by_name(db: Session) -> None:
    """Test retrieving a project by name and organization ID."""
    project = create_test_project(db)

    fetched_project = get_project_by_name(
        session=db, project_name=project.name, organization_id=project.organization_id
    )
    assert fetched_project is not None
    assert fetched_project.id == project.id
    assert fetched_project.name == project.name
    assert fetched_project.organization_id == project.organization_id


def test_get_project_by_name_not_found(db: Session) -> None:
    """Test retrieving a project by name when it doesn't exist."""
    organization = create_test_organization(db)

    # Try to get a project that doesn't exist
    non_existent_name = f"non-existent-{random_lower_string()}"
    fetched_project = get_project_by_name(
        session=db, project_name=non_existent_name, organization_id=organization.id
    )
    assert fetched_project is None


def test_get_projects_by_organization(db: Session) -> None:
    """Test retrieving all projects for an organization."""
    organization = create_test_organization(db)

    project_1 = create_project(
        session=db,
        project_create=ProjectCreate(
            name="Project 1",
            description="Test project 1",
            is_active=True,
            organization_id=organization.id,
        ),
    )

    project_2 = create_project(
        session=db,
        project_create=ProjectCreate(
            name="Project 2",
            description="Test project 2",
            is_active=True,
            organization_id=organization.id,
        ),
    )

    projects = get_projects_by_organization(session=db, org_id=organization.id)

    assert project_1 in projects
    assert project_2 in projects


def test_get_non_existent_project(db: Session) -> None:
    non_existent_project_id = get_non_existent_id(db, Project)
    fetched_project = get_project_by_id(session=db, project_id=non_existent_project_id)
    assert fetched_project is None


def test_validate_project_success(db: Session) -> None:
    """Test that a valid and active project passes validation."""
    project = create_test_project(db)

    validated_project = validate_project(session=db, project_id=project.id)
    assert validated_project.id == project.id


def test_validate_project_not_found(db: Session) -> None:
    """Test that validation fails when project does not exist."""
    non_existent_project_id = get_non_existent_id(db, Project)
    with pytest.raises(HTTPException, match="Project not found"):
        validate_project(session=db, project_id=non_existent_project_id)


def test_update_project_settings_merges_and_preserves_existing(db: Session) -> None:
    """A patched key is overwritten while untouched keys survive the merge."""
    project = create_test_project(db)
    project.settings = {"existing_key": "keep", "tracing": False}
    db.add(project)
    db.commit()

    updated = update_project_settings(
        session=db, project_id=project.id, settings_patch={"tracing": True}
    )

    assert updated.settings == {"existing_key": "keep", "tracing": True}


def test_update_project_settings_not_found(db: Session) -> None:
    non_existent_project_id = get_non_existent_id(db, Project)
    with pytest.raises(HTTPException, match="Project not found"):
        update_project_settings(
            session=db,
            project_id=non_existent_project_id,
            settings_patch={"tracing": True},
        )


def test_is_tracing_enabled_true_when_opted_in(db: Session) -> None:
    project = create_test_project(db)
    project.settings = {"tracing": True}
    db.add(project)
    db.commit()

    assert is_tracing_enabled(session=db, project_id=project.id) is True


def test_is_tracing_enabled_false_when_disabled(db: Session) -> None:
    project = create_test_project(db)
    project.settings = {"tracing": False}
    db.add(project)
    db.commit()

    assert is_tracing_enabled(session=db, project_id=project.id) is False


def test_is_tracing_enabled_false_by_default(db: Session) -> None:
    """No 'tracing' key => off by default."""
    project = create_test_project(db)
    project.settings = {}
    db.add(project)
    db.commit()

    assert is_tracing_enabled(session=db, project_id=project.id) is False


def test_is_tracing_enabled_false_when_project_missing(db: Session) -> None:
    assert (
        is_tracing_enabled(session=db, project_id=get_non_existent_id(db, Project))
        is False
    )


def test_is_tracing_enabled_false_on_invalid_project_id(db: Session) -> None:
    assert is_tracing_enabled(session=db, project_id="not-an-int") is False


def test_validate_project_inactive(db: Session) -> None:
    """Test that validation fails when project is inactive."""
    organization = create_test_organization(db)

    inactive_project = create_project(
        session=db,
        project_create=ProjectCreate(
            name=random_lower_string(),
            description="Inactive project",
            is_active=False,
            organization_id=organization.id,
        ),
    )

    with pytest.raises(HTTPException, match="Project is not active"):
        validate_project(session=db, project_id=inactive_project.id)
