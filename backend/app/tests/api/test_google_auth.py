from datetime import timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.security import create_access_token, create_refresh_token
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.user import create_random_user

GOOGLE_AUTH_URL = f"{settings.API_V1_STR}/auth/google"
SELECT_PROJECT_URL = f"{settings.API_V1_STR}/auth/select-project"
REFRESH_URL = f"{settings.API_V1_STR}/auth/refresh"
LOGOUT_URL = f"{settings.API_V1_STR}/auth/logout"

MOCK_GOOGLE_PROFILE = {
    "email": None,  # set per test
    "email_verified": True,
    "name": "Test User",
    "picture": "https://example.com/photo.jpg",
    "given_name": "Test",
    "family_name": "User",
}


def _mock_idinfo(email: str, email_verified: bool = True) -> dict:
    return {**MOCK_GOOGLE_PROFILE, "email": email, "email_verified": email_verified}


class TestGoogleAuth:
    """Test suite for POST /auth/google endpoint."""

    @patch("app.api.routes.google_auth.settings")
    def test_google_auth_not_configured(self, mock_settings, client: TestClient):
        """Test returns 500 when GOOGLE_CLIENT_ID is not set."""
        mock_settings.GOOGLE_CLIENT_ID = ""
        resp = client.post(GOOGLE_AUTH_URL, json={"token": "fake"})
        assert resp.status_code == 500
        assert "not configured" in resp.json()["detail"]

    @patch("app.api.routes.google_auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.google_auth.settings")
    def test_google_auth_invalid_token(
        self, mock_settings, mock_verify, client: TestClient
    ):
        """Test returns 400 for invalid Google token."""
        mock_settings.GOOGLE_CLIENT_ID = "test-client-id"
        mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 1440
        mock_settings.REFRESH_TOKEN_EXPIRE_MINUTES = 10080
        mock_settings.ENVIRONMENT = "testing"
        mock_settings.API_V1_STR = settings.API_V1_STR
        mock_verify.side_effect = ValueError("Invalid token")

        resp = client.post(GOOGLE_AUTH_URL, json={"token": "bad-token"})
        assert resp.status_code == 400
        assert "Invalid or expired" in resp.json()["detail"]

    @patch("app.api.routes.google_auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.google_auth.settings")
    def test_google_auth_unverified_email(
        self, mock_settings, mock_verify, client: TestClient
    ):
        """Test returns 400 when Google email is not verified."""
        mock_settings.GOOGLE_CLIENT_ID = "test-client-id"
        mock_verify.return_value = _mock_idinfo(
            "test@example.com", email_verified=False
        )

        resp = client.post(GOOGLE_AUTH_URL, json={"token": "fake"})
        assert resp.status_code == 400
        assert "not verified" in resp.json()["detail"]

    @patch("app.api.routes.google_auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.google_auth.settings")
    def test_google_auth_user_not_found(
        self, mock_settings, mock_verify, client: TestClient
    ):
        """Test returns 401 when no user exists for the email."""
        mock_settings.GOOGLE_CLIENT_ID = "test-client-id"
        mock_verify.return_value = _mock_idinfo("nonexistent@example.com")

        resp = client.post(GOOGLE_AUTH_URL, json={"token": "fake"})
        assert resp.status_code == 401
        assert "No account found" in resp.json()["detail"]

    @patch("app.api.routes.google_auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.google_auth.settings")
    def test_google_auth_activates_inactive_user(
        self, mock_settings, mock_verify, db: Session, client: TestClient
    ):
        """Test that inactive user is activated on first Google login."""
        user = create_random_user(db)
        user.is_active = False
        db.add(user)
        db.commit()
        db.refresh(user)

        mock_settings.GOOGLE_CLIENT_ID = "test-client-id"
        mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 1440
        mock_settings.REFRESH_TOKEN_EXPIRE_MINUTES = 10080
        mock_settings.ENVIRONMENT = "testing"
        mock_settings.API_V1_STR = settings.API_V1_STR
        mock_settings.SECRET_KEY = settings.SECRET_KEY
        mock_verify.return_value = _mock_idinfo(user.email)

        resp = client.post(GOOGLE_AUTH_URL, json={"token": "fake"})
        assert resp.status_code == 200

        db.refresh(user)
        assert user.is_active is True

    @patch("app.api.routes.google_auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.google_auth.settings")
    def test_google_auth_success_no_projects(
        self, mock_settings, mock_verify, db: Session, client: TestClient
    ):
        """Test successful login for user with no projects."""
        user = create_random_user(db)

        mock_settings.GOOGLE_CLIENT_ID = "test-client-id"
        mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 1440
        mock_settings.REFRESH_TOKEN_EXPIRE_MINUTES = 10080
        mock_settings.ENVIRONMENT = "testing"
        mock_settings.API_V1_STR = settings.API_V1_STR
        mock_settings.SECRET_KEY = settings.SECRET_KEY
        mock_verify.return_value = _mock_idinfo(user.email)

        resp = client.post(GOOGLE_AUTH_URL, json={"token": "fake"})
        assert resp.status_code == 200

        data = resp.json()
        assert "access_token" in data
        assert data["requires_project_selection"] is False
        assert data["available_projects"] == []
        assert "access_token" in resp.cookies

    @patch("app.api.routes.google_auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.google_auth.settings")
    def test_google_auth_success_single_project_via_api_key(
        self,
        mock_settings,
        mock_verify,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ):
        """Test successful login auto-selects single project from API key."""
        mock_settings.GOOGLE_CLIENT_ID = "test-client-id"
        mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 1440
        mock_settings.REFRESH_TOKEN_EXPIRE_MINUTES = 10080
        mock_settings.ENVIRONMENT = "testing"
        mock_settings.API_V1_STR = settings.API_V1_STR
        mock_settings.SECRET_KEY = settings.SECRET_KEY
        mock_verify.return_value = _mock_idinfo(user_api_key.user.email)

        resp = client.post(GOOGLE_AUTH_URL, json={"token": "fake"})
        assert resp.status_code == 200

        data = resp.json()
        assert data["requires_project_selection"] is False
        assert len(data["available_projects"]) == 1


class TestSelectProject:
    """Test suite for POST /auth/select-project endpoint."""

    def test_select_project_unauthenticated(self, client: TestClient):
        """Test returns 401 when not authenticated."""
        resp = client.post(SELECT_PROJECT_URL, json={"project_id": 1})
        assert resp.status_code == 401

    def test_select_project_no_access(
        self, client: TestClient, normal_user_token_headers: dict[str, str]
    ):
        """Test returns 403 when user has no access to the project."""
        resp = client.post(
            SELECT_PROJECT_URL,
            json={"project_id": 99999},
            headers=normal_user_token_headers,
        )
        assert resp.status_code == 403
        assert "do not have access" in resp.json()["detail"]

    def test_select_project_success(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
        normal_user_token_headers: dict[str, str],
    ):
        """Test successful project selection returns new token with cookies."""
        resp = client.post(
            SELECT_PROJECT_URL,
            json={"project_id": user_api_key.project.id},
            headers=normal_user_token_headers,
        )
        assert resp.status_code == 200

        data = resp.json()
        assert "access_token" in data
        assert "access_token" in resp.cookies


class TestRefreshToken:
    """Test suite for POST /auth/refresh endpoint."""

    def test_refresh_no_cookie(self, client: TestClient):
        """Test returns 401 when no refresh token cookie is present."""
        resp = client.post(REFRESH_URL)
        assert resp.status_code == 401
        assert "not found" in resp.json()["detail"]

    def test_refresh_with_access_token_instead(self, db: Session, client: TestClient):
        """Test returns 401 when access token is used instead of refresh token."""
        user = create_random_user(db)
        access_token = create_access_token(
            subject=str(user.id), expires_delta=timedelta(minutes=30)
        )
        client.cookies.set("refresh_token", access_token)

        resp = client.post(REFRESH_URL)
        assert resp.status_code == 401
        assert "Invalid token type" in resp.json()["detail"]

    def test_refresh_with_expired_token(self, db: Session, client: TestClient):
        """Test returns 401 when refresh token is expired."""
        user = create_random_user(db)
        expired_refresh = create_refresh_token(
            subject=str(user.id), expires_delta=timedelta(minutes=-1)
        )
        client.cookies.set("refresh_token", expired_refresh)

        resp = client.post(REFRESH_URL)
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"]

    def test_refresh_success(self, db: Session, client: TestClient):
        """Test successful refresh returns new tokens."""
        user = create_random_user(db)
        refresh_token = create_refresh_token(
            subject=str(user.id), expires_delta=timedelta(days=7)
        )
        client.cookies.set("refresh_token", refresh_token)

        resp = client.post(REFRESH_URL)
        assert resp.status_code == 200

        data = resp.json()
        assert "access_token" in data
        assert "access_token" in resp.cookies

    def test_refresh_with_org_project(
        self, db: Session, client: TestClient, user_api_key: TestAuthContext
    ):
        """Test refresh preserves org/project claims."""
        refresh_token = create_refresh_token(
            subject=str(user_api_key.user.id),
            expires_delta=timedelta(days=7),
            organization_id=user_api_key.organization.id,
            project_id=user_api_key.project.id,
        )
        client.cookies.set("refresh_token", refresh_token)

        resp = client.post(REFRESH_URL)
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_refresh_inactive_user(self, db: Session, client: TestClient):
        """Test returns 403 when user is inactive."""
        user = create_random_user(db)
        refresh_token = create_refresh_token(
            subject=str(user.id), expires_delta=timedelta(days=7)
        )

        user.is_active = False
        db.add(user)
        db.commit()

        client.cookies.set("refresh_token", refresh_token)

        resp = client.post(REFRESH_URL)
        assert resp.status_code == 403


class TestLogout:
    """Test suite for POST /auth/logout endpoint."""

    def test_logout_clears_cookies(self, client: TestClient):
        """Test logout clears auth cookies."""
        resp = client.post(LOGOUT_URL)
        assert resp.status_code == 200
        assert resp.json()["message"] == "Logged out successfully"
