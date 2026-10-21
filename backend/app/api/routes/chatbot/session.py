from typing import Annotated
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.celery.utils import start_chatbot_session_message
from app.crud.chatbot import (
    create_chatbot_session,
    create_chatbot_session_message,
    get_chatbot_session,
    list_chatbot_session_messages,
    list_chatbot_sessions_by_contact_ref,
)
from app.models import (
    CONTACT_REF_MAX_LENGTH,
    ChatbotMessageRoleEnum,
    ChatbotSessionSummary,
    CreateChatbotSessionRequest,
    CreateChatbotSessionResponse,
    SendChatbotSessionMessageRequest,
    SendChatbotSessionMessageResponse,
)
from app.services.chatbot.config import resolve_chatbot_model_settings
from app.services.chatbot.exceptions import ChatbotInputError
from app.utils import APIResponse, load_description, validate_callback_url

DEFAULT_TRACE_ID = "N/A"

router = APIRouter()


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


@router.get(
    "/sessions",
    description=load_description("chatbot/list_sessions.md"),
    response_model=APIResponse[list[ChatbotSessionSummary]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_chatbot_sessions(
    contact_ref: Annotated[
        str,
        Query(
            min_length=1,
            max_length=CONTACT_REF_MAX_LENGTH,
            description="Caller-side identifier of the end user whose sessions to list",
        ),
    ],
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[list[ChatbotSessionSummary]]:
    chatbot_sessions = list_chatbot_sessions_by_contact_ref(
        session,
        contact_ref=contact_ref,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    summaries: list[ChatbotSessionSummary] = []
    for chatbot_session in chatbot_sessions:
        summary = ChatbotSessionSummary(
            session_id=chatbot_session.id,
            contact_ref=chatbot_session.contact_ref,
            status=chatbot_session.status,
            collected_fields=chatbot_session.collected_fields,
            config_id=chatbot_session.config_id,
            config_version=chatbot_session.config_version,
            inserted_at=chatbot_session.inserted_at,
            updated_at=chatbot_session.updated_at,
        )
        summaries.append(summary)
    return APIResponse.success_response(summaries)


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

    start_chatbot_session_message(
        session_id=str(session_id),
        organization_id=organization_id,
        project_id=project_id,
        webhook_url=request.webhook_url,
        trace_id=correlation_id.get() or DEFAULT_TRACE_ID,
    )
    return APIResponse.success_response(SendChatbotSessionMessageResponse())
