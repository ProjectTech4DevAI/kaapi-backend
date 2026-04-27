"""Feature flags endpoint — returns resolved flags for the caller's scope."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.feature_flags import parse_feature_flag, resolve_all_flags
from app.crud.feature_flag import (
    create_feature_flag,
    delete_feature_flag,
    list_feature_flags,
    update_feature_flag,
)
from app.crud.organization import validate_organization
from app.crud.project import validate_project
from app.models import (
    FeatureFlagCreate,
    FeatureFlagDelete,
    FeatureFlagPublic,
    FeatureFlagUpdate,
)

router = APIRouter(tags=["Features"])


def _parse_flag_or_400(flag_key: str):
    try:
        return parse_feature_flag(flag_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/features",
    response_model=dict[str, bool],
    dependencies=[Depends(require_permission(Permission.REQUIRE_ORGANIZATION))],
)
def get_features(session: SessionDep, auth_context: AuthContextDep) -> dict[str, bool]:
    """Return all feature flags resolved for the caller's org + project."""
    org_id = auth_context.organization_.id
    project_id = auth_context.project.id if auth_context.project else None
    return resolve_all_flags(
        session=session,
        organization_id=org_id,
        project_id=project_id,
    )


@router.get(
    "/feature-flags",
    response_model=list[FeatureFlagPublic],
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def get_feature_flags(
    session: SessionDep,
    key: str | None = None,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> list[FeatureFlagPublic]:
    normalized_key = None
    if key:
        normalized_key = _parse_flag_or_400(key).value
    if project_id is not None and organization_id is None:
        raise HTTPException(
            status_code=400,
            detail="organization_id is required when project_id is set",
        )
    if organization_id is not None:
        validate_organization(session=session, org_id=organization_id)
    if project_id is not None:
        project = validate_project(session=session, project_id=project_id)
        if project.organization_id != organization_id:
            raise HTTPException(
                status_code=400,
                detail="project_id does not belong to organization_id",
            )
    return list_feature_flags(
        session=session,
        key=normalized_key,
        organization_id=organization_id,
        project_id=project_id,
    )


@router.post(
    "/feature-flags",
    response_model=FeatureFlagPublic,
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def create_feature_flag_route(
    session: SessionDep,
    payload: FeatureFlagCreate,
) -> FeatureFlagPublic:
    flag = _parse_flag_or_400(payload.key)
    validate_organization(session=session, org_id=payload.organization_id)
    if payload.project_id is not None:
        project = validate_project(session=session, project_id=payload.project_id)
        if project.organization_id != payload.organization_id:
            raise HTTPException(
                status_code=400,
                detail="project_id does not belong to organization_id",
            )
    created = create_feature_flag(
        session=session,
        key=flag.value,
        organization_id=payload.organization_id,
        project_id=payload.project_id,
        enabled=payload.enabled,
    )
    if created is None:
        raise HTTPException(status_code=409, detail="Feature flag already exists")
    return created


@router.patch(
    "/feature-flags",
    response_model=FeatureFlagPublic,
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def patch_feature_flag(
    session: SessionDep,
    payload: FeatureFlagUpdate,
) -> FeatureFlagPublic:
    flag = _parse_flag_or_400(payload.key)
    validate_organization(session=session, org_id=payload.organization_id)
    if payload.project_id is not None:
        project = validate_project(session=session, project_id=payload.project_id)
        if project.organization_id != payload.organization_id:
            raise HTTPException(
                status_code=400,
                detail="project_id does not belong to organization_id",
            )
    updated = update_feature_flag(
        session=session,
        key=flag.value,
        organization_id=payload.organization_id,
        project_id=payload.project_id,
        enabled=payload.enabled,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Feature flag not found")
    return updated


@router.delete(
    "/feature-flags",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def remove_feature_flag(
    session: SessionDep,
    payload: FeatureFlagDelete,
) -> dict[str, bool]:
    flag = _parse_flag_or_400(payload.key)
    validate_organization(session=session, org_id=payload.organization_id)
    if payload.project_id is not None:
        project = validate_project(session=session, project_id=payload.project_id)
        if project.organization_id != payload.organization_id:
            raise HTTPException(
                status_code=400,
                detail="project_id does not belong to organization_id",
            )
    deleted = delete_feature_flag(
        session=session,
        key=flag.value,
        organization_id=payload.organization_id,
        project_id=payload.project_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Feature flag not found")
    return {"deleted": True}
