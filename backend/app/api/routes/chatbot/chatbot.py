from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.chatbot import (
    create_chatbot,
    delete_chatbot,
    get_chatbot,
    list_chatbots,
    update_chatbot,
)
from app.models import Chatbot, ChatbotCreate, ChatbotPublic, ChatbotUpdate, Message
from app.services.chatbot.flow import CHATBOT_NOT_FOUND_MESSAGE
from app.utils import APIResponse, load_description

CHATBOT_DELETED_MESSAGE = "Chatbot deleted successfully"

router = APIRouter()


def _get_chatbot_or_404(
    session: SessionDep, auth_context: AuthContextDep, chatbot_id: UUID
) -> Chatbot:
    chatbot = get_chatbot(
        session,
        chatbot_id=chatbot_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    if chatbot is None:
        raise HTTPException(status_code=404, detail=CHATBOT_NOT_FOUND_MESSAGE)
    return chatbot


@router.post(
    "",
    status_code=201,
    description=load_description("chatbot/create.md"),
    response_model=APIResponse[ChatbotPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_chatbot_route(
    request: ChatbotCreate,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[ChatbotPublic]:
    chatbot = create_chatbot(
        session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        name=request.name,
    )
    return APIResponse.success_response(ChatbotPublic.model_validate(chatbot))


@router.get(
    "",
    description=load_description("chatbot/list.md"),
    response_model=APIResponse[list[ChatbotPublic]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_chatbots_route(
    session: SessionDep,
    auth_context: AuthContextDep,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[
        int, Query(ge=1, le=100, description="Maximum records to return")
    ] = 100,
) -> APIResponse[list[ChatbotPublic]]:
    chatbots = list_chatbots(
        session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        skip=skip,
        limit=limit,
    )
    chatbots_public: list[ChatbotPublic] = []
    for chatbot in chatbots:
        chatbots_public.append(ChatbotPublic.model_validate(chatbot))
    return APIResponse.success_response(chatbots_public)


@router.get(
    "/{chatbot_id}",
    description=load_description("chatbot/get.md"),
    response_model=APIResponse[ChatbotPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_chatbot_route(
    chatbot_id: Annotated[UUID, Path(description="Chatbot identifier")],
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[ChatbotPublic]:
    chatbot = _get_chatbot_or_404(session, auth_context, chatbot_id)
    return APIResponse.success_response(ChatbotPublic.model_validate(chatbot))


@router.patch(
    "/{chatbot_id}",
    description=load_description("chatbot/update.md"),
    response_model=APIResponse[ChatbotPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def update_chatbot_route(
    chatbot_id: Annotated[UUID, Path(description="Chatbot identifier")],
    request: ChatbotUpdate,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[ChatbotPublic]:
    chatbot = _get_chatbot_or_404(session, auth_context, chatbot_id)
    chatbot = update_chatbot(session, chatbot=chatbot, chatbot_update=request)
    return APIResponse.success_response(ChatbotPublic.model_validate(chatbot))


@router.delete(
    "/{chatbot_id}",
    description=load_description("chatbot/delete.md"),
    response_model=APIResponse[Message],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def delete_chatbot_route(
    chatbot_id: Annotated[UUID, Path(description="Chatbot identifier")],
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[Message]:
    chatbot = _get_chatbot_or_404(session, auth_context, chatbot_id)
    delete_chatbot(session, chatbot=chatbot)
    return APIResponse.success_response(Message(message=CHATBOT_DELETED_MESSAGE))
