from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.webhook_log import get_webhook_log, list_webhook_logs
from app.models import (
    WebhookDeliveryStatusEnum,
    WebhookLogPublic,
    WebhookResourceTypeEnum,
)
from app.utils import APIResponse, load_description

router = APIRouter(prefix="/webhook-logs", tags=["Webhook Logs"])


@router.get(
    "",
    response_model=APIResponse[list[WebhookLogPublic]],
    description=load_description("webhook_logs/list.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_webhook_logs_route(
    session: SessionDep,
    current_user: AuthContextDep,
    resource_type: Annotated[
        WebhookResourceTypeEnum | None, Query(description="Filter by resource type")
    ] = None,
    resource_id: Annotated[
        str | None, Query(description="Filter by resource id")
    ] = None,
    status: Annotated[
        WebhookDeliveryStatusEnum | None, Query(description="Filter by outcome")
    ] = None,
    skip: Annotated[int, Query(ge=0, description="How many items to skip")] = 0,
    limit: Annotated[
        int, Query(ge=1, le=100, description="Maximum items to return")
    ] = 100,
) -> APIResponse[list[WebhookLogPublic]]:
    logs, total = list_webhook_logs(
        session=session,
        project_id=current_user.project_.id,
        organization_id=current_user.organization_.id,
        resource_type=resource_type,
        resource_id=resource_id,
        status=status,
        skip=skip,
        limit=limit,
    )
    return APIResponse.success_response(
        data=[WebhookLogPublic.model_validate(log) for log in logs],
        metadata={"skip": skip, "limit": limit, "total": total},
    )


@router.get(
    "/{webhook_log_id}",
    response_model=APIResponse[WebhookLogPublic],
    description=load_description("webhook_logs/get.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_webhook_log_route(
    webhook_log_id: Annotated[UUID, Path(description="Webhook log ID to fetch")],
    session: SessionDep,
    current_user: AuthContextDep,
) -> APIResponse[WebhookLogPublic]:
    log = get_webhook_log(
        session=session,
        webhook_log_id=webhook_log_id,
        project_id=current_user.project_.id,
        organization_id=current_user.organization_.id,
    )
    if log is None:
        raise HTTPException(status_code=404, detail="Webhook log not found")
    return APIResponse.success_response(WebhookLogPublic.model_validate(log))
