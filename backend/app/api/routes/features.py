"""Feature flags endpoint — returns resolved flags for the caller's scope."""

from fastapi import APIRouter, Depends

from app.api.deps import AuthContextDep
from app.api.permissions import Permission, require_permission
from app.core.feature_flags import resolve_all_flags

router = APIRouter(tags=["Features"])


@router.get(
    "/features",
    response_model=dict[str, bool],
    dependencies=[Depends(require_permission(Permission.REQUIRE_ORGANIZATION))],
)
def get_features(auth_context: AuthContextDep) -> dict[str, bool]:
    """Return all feature flags resolved for the caller's org + project."""
    org_id = auth_context.organization_.id
    project_id = auth_context.project.id if auth_context.project else None
    return resolve_all_flags(
        organization_id=org_id,
        project_id=project_id,
    )
