import logging

from fastapi import APIRouter, Depends

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.exception_handlers import HTTPException
from app.core.providers import validate_provider
from app.crud.credentials import (
    get_creds_by_org,
    get_provider_credential,
    remove_creds_for_org,
    remove_provider_credential,
    set_creds_for_org,
    update_creds_for_org,
)
from app.models import CredsCreate, CredsPublic, CredsUpdate
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/credentials", tags=["Credentials"])


@router.post(
    "/",
    response_model=APIResponse[list[CredsPublic]],
    description=load_description("credentials/create.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_new_credential(
    *,
    session: SessionDep,
    creds_in: CredsCreate,
    _current_user: AuthContextDep,
):
    # Project comes from API key context; no cross-org check needed here
    # Database unique constraint ensures no duplicate credentials per provider-org-project combination

    created_creds = set_creds_for_org(
        session=session,
        creds_add=creds_in,
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )
    if not created_creds:
        logger.error(
            f"[create_new_credential] Failed to create credentials | organization_id: {_current_user.organization_.id}, project_id: {_current_user.project_.id}"
        )
        raise HTTPException(status_code=500, detail="Failed to create credentials")

    return APIResponse.success_response([cred.to_public() for cred in created_creds])


@router.get(
    "/",
    response_model=APIResponse[list[CredsPublic]],
    description=load_description("credentials/list.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def read_credential(
    *,
    session: SessionDep,
    _current_user: AuthContextDep,
):
    creds = get_creds_by_org(
        session=session,
        org_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )
    if not creds:
        raise HTTPException(status_code=404, detail="Credentials not found")

    return APIResponse.success_response([cred.to_public() for cred in creds])


@router.get(
    "/provider/{provider}",
    response_model=APIResponse[dict],
    description=load_description("credentials/get_provider.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def read_provider_credential(
    *,
    session: SessionDep,
    provider: str,
    _current_user: AuthContextDep,
):
    provider_enum = validate_provider(provider)
    credential = get_provider_credential(
        session=session,
        org_id=_current_user.organization_.id,
        provider=provider_enum,
        project_id=_current_user.project_.id,
    )
    if credential is None:
        raise HTTPException(status_code=404, detail="Provider credentials not found")

    return APIResponse.success_response(credential)


@router.patch(
    "/",
    response_model=APIResponse[list[CredsPublic]],
    description=load_description("credentials/update.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def update_credential(
    *,
    session: SessionDep,
    creds_in: CredsUpdate,
    _current_user: AuthContextDep,
):
    if not creds_in or not creds_in.provider or not creds_in.credential:
        logger.error(
            f"[update_credential] Invalid input | organization_id: {_current_user.organization_.id}, project_id: {_current_user.project_.id}"
        )
        raise HTTPException(
            status_code=400, detail="Provider and credential must be provided"
        )

    # Pass project_id directly to the CRUD function since CredsUpdate no longer has this field
    updated_credential = update_creds_for_org(
        session=session,
        org_id=_current_user.organization_.id,
        creds_in=creds_in,
        project_id=_current_user.project_.id,
    )

    return APIResponse.success_response(
        [cred.to_public() for cred in updated_credential]
    )


@router.delete(
    "/provider/{provider}",
    response_model=APIResponse[dict],
    description=load_description("credentials/delete_provider.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def delete_provider_credential(
    *,
    session: SessionDep,
    provider: str,
    _current_user: AuthContextDep,
):
    provider_enum = validate_provider(provider)
    remove_provider_credential(
        session=session,
        org_id=_current_user.organization_.id,
        provider=provider_enum,
        project_id=_current_user.project_.id,
    )

    return APIResponse.success_response(
        {"message": "Provider credentials removed successfully"}
    )


@router.delete(
    "/",
    response_model=APIResponse[dict],
    description=load_description("credentials/delete_all.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def delete_all_credentials(
    *,
    session: SessionDep,
    _current_user: AuthContextDep,
):
    remove_creds_for_org(
        session=session,
        org_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )

    return APIResponse.success_response(
        {"message": "All credentials deleted successfully"}
    )
