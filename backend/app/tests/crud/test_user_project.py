from sqlmodel import Session

from app.crud.user_project import (
    add_user_to_project,
    get_user_projects,
    get_users_by_project,
    remove_user_from_project,
)
from app.models import User, UserProject
from app.tests.utils.test_data import create_test_project
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import random_email


class TestAddUserToProject:
    """Test suite for add_user_to_project CRUD function."""

    def test_add_new_user_creates_user_and_mapping(self, db: Session):
        """Test adding a new email creates user (inactive) and project mapping."""
        project = create_test_project(db)
        email = random_email()

        user, add_status = add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )

        assert add_status == "added"
        assert user.email == email
        assert user.is_active is False

    def test_add_new_user_with_full_name(self, db: Session):
        """Test adding a new user with full_name."""
        project = create_test_project(db)
        email = random_email()

        user, add_status = add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
            full_name="Test User",
        )

        assert add_status == "added"
        assert user.full_name == "Test User"

    def test_add_existing_user_updates_full_name(self, db: Session):
        """Test adding existing user without full_name updates it."""
        project = create_test_project(db)
        user = create_random_user(db)
        user.full_name = None
        db.add(user)
        db.flush()

        returned_user, add_status = add_user_to_project(
            session=db,
            email=user.email,
            organization_id=project.organization_id,
            project_id=project.id,
            full_name="Updated Name",
        )

        assert add_status == "added"
        assert returned_user.full_name == "Updated Name"

    def test_add_user_same_project_returns_same_project(self, db: Session):
        """Test adding user already in same project returns same_project."""
        project = create_test_project(db)
        email = random_email()

        add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )

        _, add_status = add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )

        assert add_status == "same_project"

    def test_add_user_different_project_returns_different_project(self, db: Session):
        """Test adding user already in another project returns different_project."""
        project1 = create_test_project(db)
        project2 = create_test_project(db)
        email = random_email()

        add_user_to_project(
            session=db,
            email=email,
            organization_id=project1.organization_id,
            project_id=project1.id,
        )

        _, add_status = add_user_to_project(
            session=db,
            email=email,
            organization_id=project2.organization_id,
            project_id=project2.id,
        )

        assert add_status == "different_project"


class TestGetUsersByProject:
    """Test suite for get_users_by_project CRUD function."""

    def test_returns_empty_for_project_with_no_users(self, db: Session):
        """Test returns empty list when no users are mapped."""
        project = create_test_project(db)
        result = get_users_by_project(session=db, project_id=project.id)
        assert result == []

    def test_returns_users_for_project(self, db: Session):
        """Test returns mapped users for a project."""
        project = create_test_project(db)
        email = random_email()

        add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )

        result = get_users_by_project(session=db, project_id=project.id)
        assert len(result) == 1
        assert result[0].email == email


class TestRemoveUserFromProject:
    """Test suite for remove_user_from_project CRUD function."""

    def test_remove_existing_mapping(self, db: Session):
        """Test removing a user from a project."""
        project = create_test_project(db)
        email = random_email()

        user, _ = add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )

        removed = remove_user_from_project(
            session=db, user_id=user.id, project_id=project.id
        )
        assert removed is True

    def test_remove_nonexistent_mapping_returns_false(self, db: Session):
        """Test removing a non-existent mapping returns False."""
        removed = remove_user_from_project(session=db, user_id=99999, project_id=99999)
        assert removed is False

    def test_remove_last_project_deactivates_user(self, db: Session):
        """Test removing user from their last project deactivates (not deletes) the user."""
        project = create_test_project(db)
        email = random_email()

        user, _ = add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )
        user_id = user.id
        # Simulate an activated user before they lose their last project.
        user.is_active = True
        db.add(user)
        db.flush()

        remove_user_from_project(session=db, user_id=user_id, project_id=project.id)

        deactivated_user = db.get(User, user_id)
        assert deactivated_user is not None
        assert deactivated_user.is_active is False

    def test_remove_last_project_preserves_superuser(self, db: Session):
        """Test superuser is not deactivated when removed from last project."""
        project = create_test_project(db)
        user = create_random_user(db)
        user.is_superuser = True
        user.is_active = True
        db.add(user)
        db.flush()

        mapping = UserProject(
            user_id=user.id,
            organization_id=project.organization_id,
            project_id=project.id,
        )
        db.add(mapping)
        db.flush()

        remove_user_from_project(session=db, user_id=user.id, project_id=project.id)

        preserved_user = db.get(User, user.id)
        assert preserved_user is not None
        assert preserved_user.is_active is True


class TestGetUserProjects:
    """Test suite for get_user_projects CRUD function."""

    def test_returns_empty_for_user_with_no_projects(self, db: Session):
        """Test returns empty when user has no project mappings."""
        user = create_random_user(db)
        result = get_user_projects(session=db, user_id=user.id)
        assert len(result) == 0

    def test_returns_projects_for_user(self, db: Session):
        """Test returns project mappings for a user."""
        project = create_test_project(db)
        email = random_email()

        user, _ = add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )

        result = get_user_projects(session=db, user_id=user.id)
        assert len(result) == 1
        assert result[0].project_id == project.id
