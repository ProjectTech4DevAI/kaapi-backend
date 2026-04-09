from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.crud.user_project import add_user_to_project
from app.models import UserProject
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_project
from app.tests.utils.utils import random_email

USER_PROJECTS_URL = f"{settings.API_V1_STR}/user-projects"


class TestListProjectUsers:
    """Test suite for GET /user-projects/"""

    def test_list_returns_empty(
        self,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Test listing users for a project with no users."""
        resp = client.get(
            f"{USER_PROJECTS_URL}/?project_id=99999",
            headers=superuser_token_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_list_returns_users(
        self,
        db: Session,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Test listing users returns mapped users."""
        project = create_test_project(db)
        email = random_email()
        add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )
        db.commit()

        resp = client.get(
            f"{USER_PROJECTS_URL}/?project_id={project.id}",
            headers=superuser_token_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["email"] == email


class TestAddProjectUsers:
    """Test suite for POST /user-projects/"""

    def test_add_user_requires_superuser(
        self,
        db: Session,
        client: TestClient,
        normal_user_token_headers: dict[str, str],
    ):
        """Test non-superuser cannot add users."""
        project = create_test_project(db)
        resp = client.post(
            f"{USER_PROJECTS_URL}/",
            json={
                "organization_id": project.organization_id,
                "project_id": project.id,
                "users": [{"email": random_email()}],
            },
            headers=normal_user_token_headers,
        )
        assert resp.status_code == 403

    def test_add_single_user(
        self,
        db: Session,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Test adding a single user."""
        project = create_test_project(db)
        email = random_email()

        resp = client.post(
            f"{USER_PROJECTS_URL}/",
            json={
                "organization_id": project.organization_id,
                "project_id": project.id,
                "users": [{"email": email, "full_name": "Test User"}],
            },
            headers=superuser_token_headers,
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert len(data) >= 1
        emails = [u["email"] for u in data]
        assert email in emails

    def test_add_duplicate_user_same_project(
        self,
        db: Session,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Test adding same user to same project returns 409."""
        project = create_test_project(db)
        email = random_email()

        # Add first time
        add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )
        db.commit()

        # Try adding again
        resp = client.post(
            f"{USER_PROJECTS_URL}/",
            json={
                "organization_id": project.organization_id,
                "project_id": project.id,
                "users": [{"email": email}],
            },
            headers=superuser_token_headers,
        )
        assert resp.status_code == 409
        assert "Already added to this project" in resp.json()["error"]

    def test_add_user_different_project_returns_409(
        self,
        db: Session,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Test adding user already in another project returns 409."""
        project1 = create_test_project(db)
        project2 = create_test_project(db)
        email = random_email()

        add_user_to_project(
            session=db,
            email=email,
            organization_id=project1.organization_id,
            project_id=project1.id,
        )
        db.commit()

        resp = client.post(
            f"{USER_PROJECTS_URL}/",
            json={
                "organization_id": project2.organization_id,
                "project_id": project2.id,
                "users": [{"email": email}],
            },
            headers=superuser_token_headers,
        )
        assert resp.status_code == 409
        assert "Already assigned to another project" in resp.json()["error"]


class TestDeleteProjectUser:
    """Test suite for DELETE /user-projects/{user_id}"""

    def test_delete_requires_superuser(
        self,
        client: TestClient,
        normal_user_token_headers: dict[str, str],
    ):
        """Test non-superuser cannot delete users."""
        resp = client.delete(
            f"{USER_PROJECTS_URL}/99999?project_id=1",
            headers=normal_user_token_headers,
        )
        assert resp.status_code == 403

    def test_delete_nonexistent_user(
        self,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Test deleting non-existent user returns 404."""
        resp = client.delete(
            f"{USER_PROJECTS_URL}/99999?project_id=99999",
            headers=superuser_token_headers,
        )
        assert resp.status_code == 404
        assert "User not found" in resp.json()["error"]

    def test_delete_user_success(
        self,
        db: Session,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Test successfully removing a user from a project."""
        project = create_test_project(db)
        email = random_email()

        user, _ = add_user_to_project(
            session=db,
            email=email,
            organization_id=project.organization_id,
            project_id=project.id,
        )
        db.commit()

        resp = client.delete(
            f"{USER_PROJECTS_URL}/{user.id}?project_id={project.id}",
            headers=superuser_token_headers,
        )
        assert resp.status_code == 200
        assert "removed" in resp.json()["data"]["message"]

    def test_cannot_delete_self(
        self,
        db: Session,
        client: TestClient,
        superuser_token_headers: dict[str, str],
        superuser_api_key: TestAuthContext,
    ):
        """Test superuser cannot remove themselves."""
        project = create_test_project(db)
        user_id = superuser_api_key.user.id

        resp = client.delete(
            f"{USER_PROJECTS_URL}/{user_id}?project_id={project.id}",
            headers=superuser_token_headers,
        )
        assert resp.status_code == 400
        assert "cannot remove yourself" in resp.json()["error"]
