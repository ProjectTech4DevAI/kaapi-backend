import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.models import User
from app.api.permissions import Permission, has_permission, require_permission
from app.api.deps import get_auth_context
from app.tests.utils.test_data import create_test_api_key


class TestHasPermission:
    """Test suite for has_permission function"""

    def test_superuser_permission_with_superuser(self, db: Session) -> None:
        """Test that superuser has SUPERUSER permission"""
        api_key_response = create_test_api_key(db)
        user = db.get(User, api_key_response.user_id)
        user.is_superuser = True
        db.add(user)
        db.commit()
        db.refresh(user)

        auth_context = get_auth_context(
            session=db, token=None, api_key=api_key_response.key
        )

        result = has_permission(auth_context, Permission.SUPERUSER, db)

        assert result is True

    def test_superuser_permission_with_regular_user(self, db: Session) -> None:
        """Test that regular user does not have SUPERUSER permission"""
        api_key_response = create_test_api_key(db)

        auth_context = get_auth_context(
            session=db, token=None, api_key=api_key_response.key
        )

        result = has_permission(auth_context, Permission.SUPERUSER, db)

        assert result is False

    def test_require_organization_permission_with_organization(
        self, db: Session
    ) -> None:
        """Test that user with organization has REQUIRE_ORGANIZATION permission"""
        api_key_response = create_test_api_key(db)

        auth_context = get_auth_context(
            session=db, token=None, api_key=api_key_response.key
        )

        result = has_permission(auth_context, Permission.REQUIRE_ORGANIZATION, db)

        assert result is True

    def test_require_organization_permission_without_organization(
        self, db: Session
    ) -> None:
        """Test that user without organization does not have REQUIRE_ORGANIZATION permission"""
        api_key_response = create_test_api_key(db)

        auth_context = get_auth_context(
            session=db, token=None, api_key=api_key_response.key
        )

        auth_context.organization = None

        result = has_permission(auth_context, Permission.REQUIRE_ORGANIZATION, db)

        assert result is False

    def test_require_project_permission_with_project(self, db: Session) -> None:
        """Test that user with project has REQUIRE_PROJECT permission"""
        api_key_response = create_test_api_key(db)

        auth_context = get_auth_context(
            session=db, token=None, api_key=api_key_response.key
        )

        result = has_permission(auth_context, Permission.REQUIRE_PROJECT, db)

        assert result is True

    def test_require_project_permission_without_project(self, db: Session) -> None:
        """Test that user without project does not have REQUIRE_PROJECT permission"""
        api_key_response = create_test_api_key(db)

        auth_context = get_auth_context(
            session=db, token=None, api_key=api_key_response.key
        )

        auth_context.project = None

        result = has_permission(auth_context, Permission.REQUIRE_PROJECT, db)

        assert result is False


class TestRequirePermission:
    """Test suite for require_permission dependency factory"""

    def test_returns_valid_permission_checker(self) -> None:
        """Test that require_permission returns a valid callable permission checker"""
        permission_checker = require_permission(Permission.SUPERUSER)

        assert callable(permission_checker)

    def test_permission_checker_passes_with_valid_permission(self, db: Session) -> None:
        """Test that permission checker passes when user has required permission"""
        api_key_response = create_test_api_key(db)
        user = db.get(User, api_key_response.user_id)
        user.is_superuser = True
        db.add(user)
        db.commit()
        db.refresh(user)
        auth_context = get_auth_context(
            session=db, token=None, api_key=api_key_response.key
        )

        permission_checker = require_permission(Permission.SUPERUSER)
        permission_checker(auth_context, db)

    def test_permission_checker_raises_403_without_permission(
        self, db: Session
    ) -> None:
        """Test that permission checker raises HTTPException with 403 when user lacks permission"""
        api_key_response = create_test_api_key(db)
        auth_context = get_auth_context(
            session=db, token=None, api_key=api_key_response.key
        )

        permission_checker = require_permission(Permission.SUPERUSER)

        with pytest.raises(HTTPException) as exc_info:
            permission_checker(auth_context, db)

        assert exc_info.value.status_code == 403


class TestPermissionEnum:
    """Test suite for Permission enum"""

    def test_permission_enum_values(self) -> None:
        """Test that Permission enum has expected values"""
        assert Permission.SUPERUSER.value == "require_superuser"
        assert Permission.REQUIRE_ORGANIZATION.value == "require_organization_id"
        assert Permission.REQUIRE_PROJECT.value == "require_project_id"

    def test_permission_enum_is_string(self) -> None:
        """Test that Permission enum members are strings"""
        assert isinstance(Permission.SUPERUSER, str)
        assert isinstance(Permission.REQUIRE_ORGANIZATION, str)
        assert isinstance(Permission.REQUIRE_PROJECT, str)
