from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.config import ConfigVersionCrud
from app.models import (
    ConfigVersionItems,
    ConfigVersionPublic,
    ConfigVersionUpdate,
    Message,
)
from app.models.config.config import ConfigTag
from app.utils import APIResponse, load_description

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
    version_create: ConfigVersionUpdate,
    current_user: AuthContextDep,
    session: SessionDep,
    tag: ConfigTag = Query(
        ConfigTag.DEFAULT,
        description=(
            "Config scope. Use 'default' for general configs or 'ASSESSMENT' "
            "for assessment configs."
        ),
    ),
):
    """
    Create a new version for an existing configuration.

    Only include the fields you want to update in config_blob.
    Provider, model, and params can be changed.
    Type is inherited from existing config and cannot be changed.
    """
    version_crud = ConfigVersionCrud(
        session=session,
        project_id=current_user.project_.id,
        config_id=config_id,
        tag=tag,
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
    tag: ConfigTag = Query(
        ConfigTag.DEFAULT,
        description=(
            "Config scope. Use 'default' for general configs or 'ASSESSMENT' "
            "for assessment configs."
        ),
    ),
):
    """
    List all versions for a specific configuration.
    Ordered by version number in descending order.
    """
    version_crud = ConfigVersionCrud(
        session=session,
        project_id=current_user.project_.id,
        config_id=config_id,
        tag=tag,
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
    tag: ConfigTag = Query(
        ConfigTag.DEFAULT,
        description=(
            "Config scope. Use 'default' for general configs or 'ASSESSMENT' "
            "for assessment configs."
        ),
    ),
):
    """
    Get a specific version of a config.
    """
    version_crud = ConfigVersionCrud(
        session=session,
        project_id=current_user.project_.id,
        config_id=config_id,
        tag=tag,
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
    tag: ConfigTag = Query(
        ConfigTag.DEFAULT,
        description=(
            "Config scope. Use 'default' for general configs or 'ASSESSMENT' "
            "for assessment configs."
        ),
    ),
):
    """
    Delete a specific version of a config.
    """
    version_crud = ConfigVersionCrud(
        session=session,
        project_id=current_user.project_.id,
        config_id=config_id,
        tag=tag,
    )
    version_crud.delete_or_raise(version_number=version_number)

    return APIResponse.success_response(
        data=Message(message="Config Version deleted successfully"),
    )
