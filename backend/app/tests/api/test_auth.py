from datetime import timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.security import create_access_token, create_refresh_token
from app.services.auth import (
    generate_invite_token,
    generate_magic_link_token,
    verify_invite_token,
    verify_magic_link_token,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.user import create_random_user

GOOGLE_AUTH_URL = f"{settings.API_V1_STR}/auth/google"
SELECT_PROJECT_URL = f"{settings.API_V1_STR}/auth/select-project"
REFRESH_URL = f"{settings.API_V1_STR}/auth/refresh"
LOGOUT_URL = f"{settings.API_V1_STR}/auth/logout"
MAGIC_LINK_URL = f"{settings.API_V1_STR}/auth/magic-link"
MAGIC_LINK_VERIFY_URL = f"{settings.API_V1_STR}/auth/magic-link/verify"
INVITE_VERIFY_URL = f"{settings.API_V1_STR}/auth/invite/verify"

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

    @patch("app.api.routes.auth.settings")
    def test_google_auth_not_configured(self, mock_settings, client: TestClient):
        """Test returns 500 when GOOGLE_CLIENT_ID is not set."""
        mock_settings.GOOGLE_CLIENT_ID = ""
        resp = client.post(GOOGLE_AUTH_URL, json={"token": "fake"})
        assert resp.status_code == 500
        assert "not configured" in resp.json()["error"]

    @patch("app.api.routes.auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.auth.settings")
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
        assert "Invalid or expired" in resp.json()["error"]

    @patch("app.api.routes.auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.auth.settings")
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
        assert "not verified" in resp.json()["error"]

    @patch("app.api.routes.auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.auth.settings")
    def test_google_auth_user_not_found(
        self, mock_settings, mock_verify, client: TestClient
    ):
        """Test returns 401 when no user exists for the email."""
        mock_settings.GOOGLE_CLIENT_ID = "test-client-id"
        mock_verify.return_value = _mock_idinfo("nonexistent@example.com")

        resp = client.post(GOOGLE_AUTH_URL, json={"token": "fake"})
        assert resp.status_code == 401
        assert "No account found" in resp.json()["error"]

    @patch("app.api.routes.auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.auth.settings")
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

    @patch("app.api.routes.auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.auth.settings")
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

        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert "access_token" in data
        assert data["requires_project_selection"] is False
        assert data["available_projects"] == []
        assert "access_token" in resp.cookies

    @patch("app.api.routes.auth.id_token.verify_oauth2_token")
    @patch("app.api.routes.auth.settings")
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

        data = resp.json()["data"]
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
        assert "do not have access" in resp.json()["error"]

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

        body = resp.json()
        assert body["success"] is True
        assert "access_token" in body["data"]
        assert "access_token" in resp.cookies


class TestRefreshToken:
    """Test suite for POST /auth/refresh endpoint."""

    def test_refresh_no_cookie(self, client: TestClient):
        """Test returns 401 when no refresh token cookie is present."""
        resp = client.post(REFRESH_URL)
        assert resp.status_code == 401
        assert "not found" in resp.json()["error"]

    def test_refresh_with_access_token_instead(self, db: Session, client: TestClient):
        """Test returns 401 when access token is used instead of refresh token."""
        user = create_random_user(db)
        access_token = create_access_token(
            subject=str(user.id), expires_delta=timedelta(minutes=30)
        )
        client.cookies.set("refresh_token", access_token)

        resp = client.post(REFRESH_URL)
        assert resp.status_code == 401
        assert "Invalid token type" in resp.json()["error"]

    def test_refresh_with_expired_token(self, db: Session, client: TestClient):
        """Test returns 401 when refresh token is expired."""
        user = create_random_user(db)
        expired_refresh = create_refresh_token(
            subject=str(user.id), expires_delta=timedelta(minutes=-1)
        )
        client.cookies.set("refresh_token", expired_refresh)

        resp = client.post(REFRESH_URL)
        assert resp.status_code == 401
        assert "expired" in resp.json()["error"]

    def test_refresh_success(self, db: Session, client: TestClient):
        """Test successful refresh returns new tokens."""
        user = create_random_user(db)
        refresh_token = create_refresh_token(
            subject=str(user.id), expires_delta=timedelta(days=7)
        )
        client.cookies.set("refresh_token", refresh_token)

        resp = client.post(REFRESH_URL)
        assert resp.status_code == 200

        body = resp.json()
        assert body["success"] is True
        assert "access_token" in body["data"]
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
        assert "access_token" in resp.json()["data"]

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

    def test_logout_success(self, client: TestClient):
        """Test logout returns success response and clears cookies."""
        resp = client.post(LOGOUT_URL)
        assert resp.status_code == 200

        body = resp.json()
        assert body["success"] is True
        assert body["data"]["message"] == "Logged out successfully"


class TestMagicLink:
    """Test suite for POST /auth/magic-link endpoint."""

    @patch("app.api.routes.auth.settings")
    def test_magic_link_email_not_configured(self, mock_settings, client: TestClient):
        """Test returns 500 when email is not configured."""
        mock_settings.emails_enabled = False
        resp = client.post(MAGIC_LINK_URL, json={"email": "test@example.com"})
        assert resp.status_code == 500

    def test_magic_link_nonexistent_user(self, client: TestClient):
        """Test returns 404 for non-existent user."""
        resp = client.post(MAGIC_LINK_URL, json={"email": "nonexistent@example.com"})
        assert resp.status_code == 404
        assert "No account found" in resp.json()["error"]

    @patch("app.api.routes.auth.send_email")
    @patch("app.api.routes.auth.settings")
    def test_magic_link_inactive_user_allowed(
        self, mock_settings, mock_send, db: Session, client: TestClient
    ):
        """Test inactive user can still request magic link to reactivate."""
        user = create_random_user(db)
        user.is_active = False
        db.add(user)
        db.commit()

        mock_settings.emails_enabled = True
        mock_settings.MAGIC_LINK_TOKEN_EXPIRE_MINUTES = 15
        mock_settings.SECRET_KEY = settings.SECRET_KEY
        mock_settings.FRONTEND_HOST = "http://localhost:3000"
        mock_settings.PROJECT_NAME = "Kaapi"

        resp = client.post(MAGIC_LINK_URL, json={"email": user.email})
        assert resp.status_code == 200
        mock_send.assert_called_once()

    @patch("app.api.routes.auth.send_email")
    @patch("app.api.routes.auth.settings")
    def test_magic_link_success(
        self, mock_settings, mock_send, db: Session, client: TestClient
    ):
        """Test sends email for valid active user."""
        user = create_random_user(db)

        mock_settings.emails_enabled = True
        mock_settings.MAGIC_LINK_TOKEN_EXPIRE_MINUTES = 15
        mock_settings.SECRET_KEY = settings.SECRET_KEY
        mock_settings.FRONTEND_HOST = "http://localhost:3000"
        mock_settings.PROJECT_NAME = "Kaapi"

        resp = client.post(MAGIC_LINK_URL, json={"email": user.email})
        assert resp.status_code == 200
        assert "login link has been sent" in resp.json()["data"]["message"]
        mock_send.assert_called_once()


class TestMagicLinkVerify:
    """Test suite for GET /auth/magic-link/verify endpoint."""

    def test_verify_invalid_token(self, client: TestClient):
        """Test returns 400 for invalid token."""
        resp = client.get(f"{MAGIC_LINK_VERIFY_URL}?token=invalid.token.here")
        assert resp.status_code == 400
        assert "expired" in resp.json()["error"] or "Invalid" in resp.json()["error"]

    def test_verify_expired_token(self, db: Session, client: TestClient):
        """Test returns 400 for expired magic link token."""
        user = create_random_user(db)
        with patch("app.services.auth.settings.MAGIC_LINK_TOKEN_EXPIRE_MINUTES", -1):
            token = generate_magic_link_token(email=user.email)

        resp = client.get(f"{MAGIC_LINK_VERIFY_URL}?token={token}")
        assert resp.status_code == 400

    def test_verify_user_not_found(self, client: TestClient):
        """Test returns 404 when user doesn't exist."""
        token = generate_magic_link_token(email="ghost@example.com")
        resp = client.get(f"{MAGIC_LINK_VERIFY_URL}?token={token}")
        assert resp.status_code == 404

    def test_verify_activates_inactive_user(self, db: Session, client: TestClient):
        """Test magic link verify activates inactive user."""
        user = create_random_user(db)
        user.is_active = False
        db.add(user)
        db.commit()
        db.refresh(user)

        token = generate_magic_link_token(email=user.email)
        resp = client.get(f"{MAGIC_LINK_VERIFY_URL}?token={token}")
        assert resp.status_code == 200

        db.refresh(user)
        assert user.is_active is True

    def test_verify_success(self, db: Session, client: TestClient):
        """Test successful magic link verification logs user in."""
        user = create_random_user(db)
        token = generate_magic_link_token(email=user.email)

        resp = client.get(f"{MAGIC_LINK_VERIFY_URL}?token={token}")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert "access_token" in resp.json()["data"]
        assert "access_token" in resp.cookies


class TestInviteVerify:
    """Test suite for GET /auth/invite/verify endpoint."""

    def test_verify_invalid_token(self, client: TestClient):
        """Test returns 400 for invalid invite token."""
        resp = client.get(f"{INVITE_VERIFY_URL}?token=invalid.token")
        assert resp.status_code == 400

    def test_verify_user_not_found(
        self, client: TestClient, user_api_key: TestAuthContext
    ):
        """Test returns 404 when invited user doesn't exist."""
        token = generate_invite_token(
            email="ghost@example.com",
            organization_id=user_api_key.organization.id,
            project_id=user_api_key.project.id,
        )
        resp = client.get(f"{INVITE_VERIFY_URL}?token={token}")
        assert resp.status_code == 404

    def test_verify_activates_inactive_user(
        self, db: Session, client: TestClient, user_api_key: TestAuthContext
    ):
        """Test invite verification activates inactive user."""
        user = create_random_user(db)
        user.is_active = False
        db.add(user)
        db.commit()
        db.refresh(user)

        token = generate_invite_token(
            email=user.email,
            organization_id=user_api_key.organization.id,
            project_id=user_api_key.project.id,
        )
        resp = client.get(f"{INVITE_VERIFY_URL}?token={token}")
        assert resp.status_code == 200

        db.refresh(user)
        assert user.is_active is True
        assert "access_token" in resp.json()["data"]

    def test_verify_success_active_user(
        self, db: Session, client: TestClient, user_api_key: TestAuthContext
    ):
        """Test invite verification works for already active user."""
        user = create_random_user(db)
        token = generate_invite_token(
            email=user.email,
            organization_id=user_api_key.organization.id,
            project_id=user_api_key.project.id,
        )
        resp = client.get(f"{INVITE_VERIFY_URL}?token={token}")
        assert resp.status_code == 200
        assert "access_token" in resp.json()["data"]


class TestTokenGeneration:
    """Test suite for services/auth.py token generation functions."""

    def test_generate_and_verify_invite_token(self):
        """Test invite token roundtrip."""
        token = generate_invite_token(
            email="test@example.com", organization_id=1, project_id=2
        )
        result = verify_invite_token(token)
        assert result is not None
        assert result.email == "test@example.com"
        assert result.organization_id == 1
        assert result.project_id == 2

    def test_verify_invite_token_wrong_type(self):
        """Test invite verify rejects magic_link tokens."""
        token = generate_magic_link_token(email="test@example.com")
        result = verify_invite_token(token)
        assert result is None

    def test_generate_and_verify_magic_link_token(self):
        """Test magic link token roundtrip."""
        token = generate_magic_link_token(email="test@example.com")
        result = verify_magic_link_token(token)
        assert result == "test@example.com"

    def test_verify_magic_link_token_wrong_type(self):
        """Test magic link verify rejects invite tokens."""
        token = generate_invite_token(
            email="test@example.com", organization_id=1, project_id=1
        )
        result = verify_magic_link_token(token)
        assert result is None

    def test_verify_invalid_token_returns_none(self):
        """Test both verify functions return None for garbage tokens."""
        assert verify_invite_token("garbage") is None
        assert verify_magic_link_token("garbage") is None

    def test_verify_invite_token_invalid(self):
        """Test invite verify returns None for garbage tokens."""
        assert verify_invite_token("garbage") is None
