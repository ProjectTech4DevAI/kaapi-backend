from enum import StrEnum

from fastapi import HTTPException
from sqlmodel import Session

from app.api.deps import AuthContextDep, SessionDep
from app.core.feature_flags import FeatureFlag
from app.models import AuthContext


class Permission(StrEnum):
    """Permission types for authorization checks"""

    SUPERUSER = "require_superuser"
    REQUIRE_ORGANIZATION = "require_organization_id"
    REQUIRE_PROJECT = "require_project_id"


def has_permission(
    auth_context: AuthContext,
    permission: Permission,
    _session: Session | None = None,
) -> bool:
    """
    Check if the auth_context has the specified permission.

    Args:
        user_context: The authenticated user context
        permission: The permission to check (Permission enum)
        session: Database session (optional)

    Returns:
        bool: True if user has permission, False otherwise
    """
    match permission:
        case Permission.SUPERUSER:
            return auth_context.user.is_superuser
        case Permission.REQUIRE_ORGANIZATION:
            return auth_context.organization is not None
        case Permission.REQUIRE_PROJECT:
            return auth_context.project is not None
        case _:
            return False


def require_permission(permission: Permission):
    """
    Dependency factory for requiring specific permissions in FastAPI routes.

    Usage:
        @app.get("/endpoint", dependencies=[Depends(require_permission(Permission.REQUIRE_ORGANIZATION))])
        def endpoint(auth_context: Annotated[AuthContext, Depends(get_user_context)]):
            pass
    """

    def permission_checker(
        auth_context: AuthContextDep,
        session: SessionDep,
    ):
        if not has_permission(auth_context, permission, session):
            error_messages = {
                Permission.SUPERUSER: "Insufficient permissions - require superuser access.",
                Permission.REQUIRE_ORGANIZATION: "Insufficient permissions - require organization access.",
                Permission.REQUIRE_PROJECT: "Insufficient permissions - require project access.",
            }
            raise HTTPException(
                status_code=403,
                detail=error_messages.get(permission, "Insufficient permissions"),
            )

    return permission_checker


def require_feature(flag: FeatureFlag):
    """Dependency factory that gates a route behind a feature flag.

    Returns 403 when the flag is disabled for the caller's org/project,
    so callers can explicitly handle feature access denial.

    Usage::

        router = APIRouter(
            dependencies=[Depends(require_feature(FeatureFlag.ASSESSMENT))]
        )
    """

    def _check_with_session(auth_context: AuthContextDep, session: SessionDep):
        from app.core.feature_flags import is_enabled

        org_id = auth_context.organization.id if auth_context.organization else None
        project_id = auth_context.project.id if auth_context.project else None

        if org_id is None or project_id is None or not is_enabled(
            session=session,
            flag=flag.value,
            organization_id=org_id,
            project_id=project_id,
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Feature '{flag.value}' is not enabled for this org or project.",
            )

    return _check_with_session
