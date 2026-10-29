from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import ChatbotFlowUpdate, ChatbotFlowVersionPublic
from app.services.chatbot.exceptions import (
    ChatbotFlowConflictError,
    ChatbotFlowNotFoundError,
    ChatbotFlowValidationError,
    ChatbotNotFoundError,
)
from app.services.chatbot.flow import (
    build_chatbot_flow_version_public,
    get_chatbot_flow,
    publish_chatbot_flow,
    save_chatbot_flow_draft,
)
from app.utils import APIResponse, load_description

router = APIRouter()


@router.get(
    "/{chatbot_id}/flow",
    description=load_description("chatbot/get_flow.md"),
    response_model=APIResponse[ChatbotFlowVersionPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_chatbot_flow_route(
    chatbot_id: Annotated[UUID, Path(description="Chatbot identifier")],
    session: SessionDep,
    auth_context: AuthContextDep,
    version: Annotated[
        int | None,
        Query(
            ge=1,
            description="Flow version to fetch; omit to get the draft if one exists, else the live version",
        ),
    ] = None,
) -> APIResponse[ChatbotFlowVersionPublic]:
    try:
        flow_version = get_chatbot_flow(
            session=session,
            chatbot_id=chatbot_id,
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            version=version,
        )
    except (ChatbotNotFoundError, ChatbotFlowNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return APIResponse.success_response(build_chatbot_flow_version_public(flow_version))


@router.put(
    "/{chatbot_id}/flow",
    description=load_description("chatbot/save_flow.md"),
    response_model=APIResponse[ChatbotFlowVersionPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def save_chatbot_flow_route(
    chatbot_id: Annotated[UUID, Path(description="Chatbot identifier")],
    request: ChatbotFlowUpdate,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[ChatbotFlowVersionPublic]:
    try:
        flow_version = save_chatbot_flow_draft(
            session=session,
            chatbot_id=chatbot_id,
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            flow=request,
        )
    except ChatbotNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChatbotFlowConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ChatbotFlowValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return APIResponse.success_response(build_chatbot_flow_version_public(flow_version))


@router.post(
    "/{chatbot_id}/flow/publish",
    description=load_description("chatbot/publish_flow.md"),
    response_model=APIResponse[ChatbotFlowVersionPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def publish_chatbot_flow_route(
    chatbot_id: Annotated[UUID, Path(description="Chatbot identifier")],
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[ChatbotFlowVersionPublic]:
    try:
        flow_version = publish_chatbot_flow(
            session=session,
            chatbot_id=chatbot_id,
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
        )
    except ChatbotNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChatbotFlowConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ChatbotFlowValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return APIResponse.success_response(build_chatbot_flow_version_public(flow_version))
