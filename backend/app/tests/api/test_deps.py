import pytest
from sqlmodel import Session
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.deps import get_auth_context
from app.models import (
    User,
    AuthContext,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.user import authentication_token_from_email, create_random_user
from app.core.config import settings
from app.tests.utils.test_data import create_test_api_key


class TestGetAuthContext:
    """Test suite for get_auth_context function"""

    def test_get_auth_context_with_valid_api_key(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Test successful authentication with valid API key"""
        auth_context = get_auth_context(
            session=db,
            token=None,
            api_key=user_api_key.key,
        )

        assert isinstance(auth_context, AuthContext)
        assert auth_context.user == user_api_key.user
        assert auth_context.project == user_api_key.project
        assert auth_context.organization == user_api_key.organization

    def test_get_auth_context_with_invalid_api_key(self, db: Session) -> None:
        """Test authentication fails with invalid API key"""
        invalid_api_key = "ApiKey InvalidKeyThatDoesNotExist123456789"

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(
                session=db,
                token=None,
                api_key=invalid_api_key,
            )

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid API Key"

    def test_get_auth_context_with_valid_token(
        self, db: Session, normal_user_token_headers: dict[str, str]
    ) -> None:
        """Test successful authentication with valid token"""
        token = normal_user_token_headers["Authorization"].replace("Bearer ", "")
        auth_context = get_auth_context(
            session=db,
            token=token,
            api_key=None,
        )

        # Assert
        assert isinstance(auth_context, AuthContext)
        assert auth_context.user.email == settings.EMAIL_TEST_USER

    def test_get_auth_context_with_invalid_token(self, db: Session) -> None:
        """Test authentication fails with invalid token"""
        invalid_token = "invalid.token"

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(
                session=db,
                token=invalid_token,
                api_key=None,
            )

        assert exc_info.value.status_code == 403

    def test_get_auth_context_with_no_credentials(self, db: Session) -> None:
        """Test authentication fails when neither API key nor token is provided"""
        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(
                session=db,
                token=None,
                api_key=None,
            )

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid Authorization format"

    def test_get_auth_context_with_inactive_user_via_api_key(self, db: Session) -> None:
        """Test authentication fails when API key belongs to inactive user"""
        api_key = create_test_api_key(db)

        user = db.get(User, api_key.user_id)
        user.is_active = False
        db.add(user)
        db.commit()
        db.refresh(user)

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(
                session=db,
                token=None,
                api_key=api_key.key,
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Inactive user"

    def test_get_auth_context_with_inactive_user_via_token(
        self, db: Session, client: TestClient
    ) -> None:
        """Test authentication fails when token belongs to inactive user"""
        user = create_random_user(db)
        token_headers = authentication_token_from_email(
            client=client, email=user.email, db=db
        )
        token = token_headers["Authorization"].replace("Bearer ", "")

        user.is_active = False
        db.add(user)
        db.commit()

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(
                session=db,
                token=token,
                api_key=None,
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Inactive user"

    def test_get_auth_context_with_inactive_organization(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Test authentication fails when organization is inactive"""
        organization = user_api_key.organization
        organization.is_active = False
        db.add(organization)
        db.commit()
        db.refresh(organization)

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(
                session=db,
                token=None,
                api_key=user_api_key.key,
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Inactive Organization"

    def test_get_auth_context_with_inactive_project(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        """Test authentication fails when project is inactive"""
        project = user_api_key.project
        project.is_active = False
        db.add(project)
        db.commit()
        db.refresh(project)

        with pytest.raises(HTTPException) as exc_info:
            get_auth_context(
                session=db,
                token=None,
                api_key=user_api_key.key,
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Inactive Project"
