from uuid import UUID
from fastapi import APIRouter, Depends, Query, HTTPException, Path

from app.api.deps import SessionDep, AuthContextDep
from app.crud.config import ConfigCrud, ConfigVersionCrud
from app.models import (
    ConfigVersionCreate,
    ConfigVersionPublic,
    Message,
    ConfigVersionItems,
)
from app.utils import APIResponse, load_description
from app.api.permissions import Permission, require_permission

router = APIRouter()


@router.post(
    "/{config_id}/versions",
    description=load_description("config/create_version.md"),
    response_model=APIResponse[ConfigVersionPublic],
    status_code=201,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_version(
    config_id: UUID,
    version_create: ConfigVersionCreate,
    current_user: AuthContextDep,
    session: SessionDep,
):
    """
    Create a new version for an existing configuration.
    The version number is automatically incremented.
    """
    version_crud = ConfigVersionCrud(
        session=session, project_id=current_user.project_.id, config_id=config_id
    )
    version = version_crud.create_or_raise(version_create=version_create)

    return APIResponse.success_response(
        data=ConfigVersionPublic(**version.model_dump()),
    )


@router.get(
    "/{config_id}/versions",
    description=load_description("config/list_versions.md"),
    response_model=APIResponse[list[ConfigVersionItems]],
    status_code=200,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_versions(
    config_id: UUID,
    current_user: AuthContextDep,
    session: SessionDep,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum records to return"),
):
    """
    List all versions for a specific configuration.
    Ordered by version number in descending order.
    """
    version_crud = ConfigVersionCrud(
        session=session, project_id=current_user.project_.id, config_id=config_id
    )
    versions = version_crud.read_all(
        skip=skip,
        limit=limit,
    )
    return APIResponse.success_response(
        data=versions,
    )


@router.get(
    "/{config_id}/versions/{version_number}",
    description=load_description("config/get_version.md"),
    response_model=APIResponse[ConfigVersionPublic],
    status_code=200,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_version(
    config_id: UUID,
    current_user: AuthContextDep,
    session: SessionDep,
    version_number: int = Path(
        ..., ge=1, description="The version number of the config"
    ),
):
    """
    Get a specific version of a config.
    """
    version_crud = ConfigVersionCrud(
        session=session, project_id=current_user.project_.id, config_id=config_id
    )
    version = version_crud.exists_or_raise(version_number=version_number)
    return APIResponse.success_response(
        data=version,
    )


@router.delete(
    "/{config_id}/versions/{version_number}",
    description=load_description("config/delete_version.md"),
    response_model=APIResponse[Message],
    status_code=200,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def delete_version(
    config_id: UUID,
    current_user: AuthContextDep,
    session: SessionDep,
    version_number: int = Path(
        ..., ge=1, description="The version number of the config"
    ),
):
    """
    Delete a specific version of a config.
    """
    version_crud = ConfigVersionCrud(
        session=session, project_id=current_user.project_.id, config_id=config_id
    )
    version_crud.delete_or_raise(version_number=version_number)

    return APIResponse.success_response(
        data=Message(message="Config Version deleted successfully"),
    )
