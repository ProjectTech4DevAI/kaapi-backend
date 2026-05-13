"""Feature flags endpoint — returns resolved flags for the caller's scope."""

import logging

from fastapi import APIRouter, Depends, Query

from app.api.deps import SessionDep
from app.api.permissions import Permission, require_permission
from app.core.feature_flags import FeatureFlag
from app.crud.feature_flag import (
    create_feature_flag,
    delete_feature_flag,
    list_feature_flags,
    update_feature_flag,
)
from app.crud.project import validate_project
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
    response_model=APIResponse[FeatureFlagPublic],
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def create_feature_flag_route(
    session: SessionDep,
    payload: FeatureFlagCreate,
) -> APIResponse[FeatureFlagPublic]:
    project = validate_project(session=session, project_id=payload.project_id)
    created = create_feature_flag(
        session=session,
        key=payload.key,
        organization_id=project.organization_id,
        project_id=payload.project_id,
        enabled=payload.enabled,
    )
    logger.info(
        "[create_feature_flag_route] Created flag=%s enabled=%s project=%s",
        payload.key,
        payload.enabled,
        payload.project_id,
    )
    return APIResponse.success_response(data=created)


@router.get(
    "/feature-flags",
    description=load_description("features/list_flags.md"),
    response_model=APIResponse[list[FeatureFlagPublic]],
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def get_feature_flags(
    session: SessionDep,
    project_id: int = Query(..., description="Project ID"),
    feature_key: FeatureFlag
    | None = Query(
        default=None,
        alias="key",
        description="Feature flag key to filter by",
    ),
) -> APIResponse[list[FeatureFlagPublic]]:
    project = validate_project(session=session, project_id=project_id)
    flags = list_feature_flags(
        session=session,
        key=feature_key.value if feature_key is not None else None,
        organization_id=project.organization_id,
        project_id=project_id,
    )
    return APIResponse.success_response(data=flags)


@router.patch(
    "/feature-flags",
    description=load_description("features/update_flag.md"),
    response_model=APIResponse[FeatureFlagPublic],
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def patch_feature_flag(
    session: SessionDep,
    payload: FeatureFlagUpdate,
) -> APIResponse[FeatureFlagPublic]:
    project = validate_project(session=session, project_id=payload.project_id)
    updated = update_feature_flag(
        session=session,
        key=payload.key,
        organization_id=project.organization_id,
        project_id=payload.project_id,
        enabled=payload.enabled,
    )
    logger.info(
        "[patch_feature_flag] Updated flag=%s enabled=%s project=%s",
        payload.key,
        payload.enabled,
        payload.project_id,
    )
    return APIResponse.success_response(data=updated)


@router.delete(
    "/feature-flags",
    description=load_description("features/delete_flag.md"),
    response_model=APIResponse[dict[str, bool]],
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
)
def remove_feature_flag(
    session: SessionDep,
    payload: FeatureFlagDelete,
) -> APIResponse[dict[str, bool]]:
    project = validate_project(session=session, project_id=payload.project_id)
    delete_feature_flag(
        session=session,
        key=payload.key,
        organization_id=project.organization_id,
        project_id=payload.project_id,
    )
    return APIResponse.success_response(data={"deleted": True})
