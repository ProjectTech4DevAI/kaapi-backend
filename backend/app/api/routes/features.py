"""Feature flags endpoint — returns resolved flags for the caller's scope."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import SessionDep
from app.api.permissions import Permission, require_permission
from app.core.feature_flags import FeatureFlag
from app.crud.feature_flag import (
    create_feature_flag,
    delete_feature_flag,
    list_feature_flags,
    update_feature_flag,
)
from app.crud.organization import validate_organization
from app.crud.project import validate_project_belongs_to_organization
from app.models import (
    FeatureFlagCreate,
    FeatureFlagDelete,
    FeatureFlagPublic,
    FeatureFlagUpdate,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Features"])

@router.post(
    "/feature-flags",
    description=load_description("features/create_flag.md"),
    response_model=FeatureFlagPublic,
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def create_feature_flag_route(
    session: SessionDep,
    payload: FeatureFlagCreate,
) -> FeatureFlagPublic:
    validate_organization(session=session, org_id=payload.organization_id)
    validate_project_belongs_to_organization(
        session=session,
        project_id=payload.project_id,
        organization_id=payload.organization_id,
    )
    created = create_feature_flag(
        session=session,
        key=payload.key,
        organization_id=payload.organization_id,
        project_id=payload.project_id,
        enabled=payload.enabled,
    )
    if created is None:
        raise HTTPException(status_code=409, detail="Feature flag already exists")
    logger.info(
        f"[create_feature_flag_route] Created flag={payload.key} "
        f"enabled={payload.enabled} org={payload.organization_id} project={payload.project_id}"
    )
    return created


@router.get(
    "/feature-flags",
    description=load_description("features/list_flags.md"),
    response_model=list[FeatureFlagPublic],
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def get_feature_flags(
    session: SessionDep,
    feature_key: FeatureFlag | None = Query(
        default=None,
        alias="key",
        description="Feature flag key to filter by",
    ),
    organization_id: int | None = None,
    project_id: int | None = None,
) -> list[FeatureFlagPublic]:
    if project_id is not None and organization_id is None:
        raise HTTPException(
            status_code=400,
            detail="organization_id is required when project_id is set",
        )
    if organization_id is not None:
        validate_organization(session=session, org_id=organization_id)
    if project_id is not None:
        assert organization_id is not None
        validate_project_belongs_to_organization(
            session=session,
            project_id=project_id,
            organization_id=organization_id,
        )
    return list_feature_flags(
        session=session,
        key=feature_key.value if feature_key is not None else None,
        organization_id=organization_id,
        project_id=project_id,
    )

@router.patch(
    "/feature-flags",
    description=load_description("features/update_flag.md"),
    response_model=FeatureFlagPublic,
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def patch_feature_flag(
    session: SessionDep,
    payload: FeatureFlagUpdate,
) -> FeatureFlagPublic:
    validate_organization(session=session, org_id=payload.organization_id)
    validate_project_belongs_to_organization(
        session=session,
        project_id=payload.project_id,
        organization_id=payload.organization_id,
    )
    updated = update_feature_flag(
        session=session,
        key=payload.key,
        organization_id=payload.organization_id,
        project_id=payload.project_id,
        enabled=payload.enabled,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Feature flag not found")
    logger.info(
        f"[patch_feature_flag] Updated flag={payload.key} "
        f"enabled={payload.enabled} org={payload.organization_id} project={payload.project_id}"
    )
    return updated


@router.delete(
    "/feature-flags",
    description=load_description("features/delete_flag.md"),
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def remove_feature_flag(
    session: SessionDep,
    payload: FeatureFlagDelete,
) -> dict[str, bool]:
    validate_organization(session=session, org_id=payload.organization_id)
    validate_project_belongs_to_organization(
        session=session,
        project_id=payload.project_id,
        organization_id=payload.organization_id,
    )
    deleted = delete_feature_flag(
        session=session,
        key=payload.key,
        organization_id=payload.organization_id,
        project_id=payload.project_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Feature flag not found")
    return {"deleted": True}
