import logging
from typing import Annotated
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Depends, HTTPException, Path

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.celery.utils import start_chatbot_session_message
from app.crud.chatbot import (
    create_chatbot_session,
    create_chatbot_session_message,
    get_chatbot_session,
    list_chatbot_session_messages,
)
from app.models import (
    ChatbotMessage,
    ChatbotMessageRoleEnum,
    ChatbotTurnRequest,
    ChatbotTurnResponse,
    CreateChatbotSessionRequest,
    CreateChatbotSessionResponse,
    SendChatbotSessionMessageRequest,
    SendChatbotSessionMessageResponse,
)
from app.services.chatbot.engine import (
    ChatbotInputError,
    ChatbotProviderError,
    run_chatbot_turn,
)
from app.services.chatbot.config import resolve_chatbot_model_settings
from app.utils import APIResponse, load_description, validate_callback_url

logger = logging.getLogger(__name__)

DEFAULT_TRACE_ID = "N/A"

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


@router.post(
    "/message",
    description=load_description("chatbot/message.md"),
    response_model=APIResponse[ChatbotTurnResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def send_chatbot_message(
    request: ChatbotTurnRequest,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[ChatbotTurnResponse]:
    history: list[dict[str, str]] = []
    for entry in request.history:
        history.append(entry.model_dump(mode="json"))

    try:
        model_settings = resolve_chatbot_model_settings(
            session=session,
            project_id=auth_context.project_.id,
            config_id=request.config_id,
            config_version=request.config_version,
            max_tokens=request.max_tokens,
            history_token_limit=request.history_token_limit,
        )
        transcript = await run_chatbot_turn(
            system_prompt=request.system_prompt,
            history=history,
            user_message=request.message,
            model_settings=model_settings,
        )
    except ChatbotInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ChatbotProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))

    messages: list[ChatbotMessage] = []
    for entry in transcript:
        messages.append(ChatbotMessage.model_validate(entry))
    return APIResponse.success_response(ChatbotTurnResponse(messages=messages))


@router.post(
    "/sessions",
    status_code=201,
    description=load_description("chatbot/create_session.md"),
    response_model=APIResponse[CreateChatbotSessionResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def start_chatbot_session(
    request: CreateChatbotSessionRequest,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[CreateChatbotSessionResponse]:
    project_id = auth_context.project_.id
    # Fail fast on an unusable config instead of on the first queued turn.
    try:
        resolve_chatbot_model_settings(
            session=session,
            project_id=project_id,
            config_id=request.config_id,
            config_version=request.config_version,
            max_tokens=None,
            history_token_limit=None,
        )
    except ChatbotInputError as e:
        raise HTTPException(status_code=422, detail=str(e))

    chatbot_session = create_chatbot_session(
        session,
        organization_id=auth_context.organization_.id,
        project_id=project_id,
        contact_ref=request.contact_ref,
        config_id=request.config_id,
        config_version=request.config_version,
        system_prompt=request.system_prompt,
    )
    return APIResponse.success_response(
        CreateChatbotSessionResponse(session_id=chatbot_session.id)
    )


@router.post(
    "/sessions/{session_id}/message",
    status_code=202,
    description=load_description("chatbot/send_session_message.md"),
    response_model=APIResponse[SendChatbotSessionMessageResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def send_chatbot_session_message(
    session_id: Annotated[UUID, Path(description="Chatbot session identifier")],
    request: SendChatbotSessionMessageRequest,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[SendChatbotSessionMessageResponse]:
    organization_id = auth_context.organization_.id
    project_id = auth_context.project_.id

    # validate_callback_url signals rejection with ValueError, which has no
    # global handler and would otherwise surface as a 500.
    try:
        validate_callback_url(request.webhook_url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid webhook_url: {e}")

    chatbot_session = get_chatbot_session(
        session,
        session_id=session_id,
        organization_id=organization_id,
        project_id=project_id,
    )
    if chatbot_session is None:
        raise HTTPException(status_code=404, detail="Chatbot session not found")

    if request.message is None:
        # Only the opening greeting may omit `message`; reject here rather than
        # letting the worker fail and report it via webhook.
        if list_chatbot_session_messages(session, session_id=session_id):
            raise HTTPException(
                status_code=422,
                detail="`message` is required once the session has started; omit it only on the first call.",
            )
    else:
        create_chatbot_session_message(
            session,
            session_id=session_id,
            role=ChatbotMessageRoleEnum.USER,
            content=request.message,
        )

    task_id = start_chatbot_session_message(
        session_id=str(session_id),
        organization_id=organization_id,
        project_id=project_id,
        webhook_url=request.webhook_url,
        trace_id=correlation_id.get() or DEFAULT_TRACE_ID,
    )
    logger.info(
        f"[send_chatbot_session_message] Turn queued | session_id: {session_id}, "
        f"project_id: {project_id}, task_id: {task_id}, has_message: {request.message is not None}"
    )
    return APIResponse.success_response(SendChatbotSessionMessageResponse())
