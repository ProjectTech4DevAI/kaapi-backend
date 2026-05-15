from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.user_project import add_user_to_project
from app.models import UserProject
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_project
from app.tests.utils.utils import random_email

USER_PROJECTS_URL = f"{settings.API_V1_STR}/user-projects"


class TestListProjectUsers:
    """Test suite for GET /user-projects"""

    def test_list_returns_empty(
        self,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Test listing users for a project with no users."""
        resp = client.get(
            f"{USER_PROJECTS_URL}?project_id=99999",
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
            f"{USER_PROJECTS_URL}?project_id={project.id}",
            headers=superuser_token_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["email"] == email


class TestAddProjectUsers:
    """Test suite for POST /user-projects"""

    def test_add_user_requires_superuser(
        self,
        db: Session,
        client: TestClient,
        normal_user_token_headers: dict[str, str],
    ):
        """Test non-superuser cannot add users."""
        project = create_test_project(db)
        resp = client.post(
            USER_PROJECTS_URL,
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
            USER_PROJECTS_URL,
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

    @patch("app.api.routes.user_project.send_email")
    @patch("app.api.routes.user_project.settings")
    def test_add_user_sends_invite_email(
        self,
        mock_settings,
        mock_send_email,
        db: Session,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Test adding a user sends an invitation email when emails are enabled."""
        project = create_test_project(db)
        email = random_email()

        mock_settings.emails_enabled = True
        mock_settings.INVITE_TOKEN_EXPIRE_HOURS = 168
        mock_settings.SECRET_KEY = settings.SECRET_KEY
        mock_settings.FRONTEND_HOST = "http://localhost:3000"
        mock_settings.PROJECT_NAME = "Kaapi"

        resp = client.post(
            USER_PROJECTS_URL,
            json={
                "organization_id": project.organization_id,
                "project_id": project.id,
                "users": [{"email": email}],
            },
            headers=superuser_token_headers,
        )
        assert resp.status_code == 201
        mock_send_email.assert_called_once()

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
            USER_PROJECTS_URL,
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
            USER_PROJECTS_URL,
            json={
                "organization_id": project2.organization_id,
                "project_id": project2.id,
                "users": [{"email": email}],
            },
            headers=superuser_token_headers,
        )
        assert resp.status_code == 409
        assert "Already assigned to another project" in resp.json()["error"]

    def test_add_bulk_surfaces_all_same_project_conflicts(
        self,
        db: Session,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """All emails already on the project should appear in the 409 error."""
        project = create_test_project(db)
        email_a = random_email()
        email_b = random_email()
        for email in (email_a, email_b):
            add_user_to_project(
                session=db,
                email=email,
                organization_id=project.organization_id,
                project_id=project.id,
            )
        db.commit()

        resp = client.post(
            f"{USER_PROJECTS_URL}/",
            json={
                "organization_id": project.organization_id,
                "project_id": project.id,
                "users": [{"email": email_a}, {"email": email_b}],
            },
            headers=superuser_token_headers,
        )
        assert resp.status_code == 409
        body = resp.json()["error"]
        assert "Already added to this project" in body
        assert email_a in body
        assert email_b in body

    def test_add_duplicate_email_in_same_request_rolls_back(
        self,
        db: Session,
        client: TestClient,
        superuser_token_headers: dict[str, str],
    ):
        """Submitting the same email twice in one request rolls back the whole batch.

        Pins current behaviour: the second occurrence is detected as a
        same-project conflict because the first occurrence was just added.
        """
        project = create_test_project(db)
        project_id = project.id
        organization_id = project.organization_id
        email = random_email()

        resp = client.post(
            f"{USER_PROJECTS_URL}/",
            json={
                "organization_id": organization_id,
                "project_id": project_id,
                "users": [{"email": email}, {"email": email}],
            },
            headers=superuser_token_headers,
        )
        assert resp.status_code == 409
        assert "Already added to this project" in resp.json()["error"]

        # Confirm rollback: no UserProject row was persisted.
        rows = db.exec(
            select(UserProject).where(UserProject.project_id == project_id)
        ).all()
        assert rows == []


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
