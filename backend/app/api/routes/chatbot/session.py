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
    get_latest_chatbot_session,
    list_chatbot_session_messages,
)
from app.models import (
    CHATBOT_REF_MAX_LENGTH,
    CONTACT_REF_MAX_LENGTH,
    ChatbotMessageRoleEnum,
    ChatbotSessionStatusEnum,
    ChatbotSessionSummary,
    CreateChatbotSessionRequest,
    CreateChatbotSessionResponse,
    SendChatbotSessionMessageRequest,
    SendChatbotSessionMessageResponse,
)
from app.services.chatbot.chatbot_ref import generate_chatbot_ref
from app.services.chatbot.config import resolve_chatbot_model_settings
from app.services.chatbot.exceptions import (
    ChatbotInputError,
    ChatbotRefGenerationError,
    ChatbotSessionConflictError,
)
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

    chatbot_ref = request.chatbot_ref
    if chatbot_ref is None:
        try:
            chatbot_ref = generate_chatbot_ref(session, project_id=project_id)
        except ChatbotRefGenerationError as e:
            raise HTTPException(status_code=503, detail=str(e))

    try:
        chatbot_session = create_chatbot_session(
            session,
            organization_id=auth_context.organization_.id,
            project_id=project_id,
            name=request.name,
            contact_ref=request.contact_ref,
            chatbot_ref=chatbot_ref,
            config_id=request.config_id,
            config_version=request.config_version,
            system_prompt=request.system_prompt,
        )
    except ChatbotSessionConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return APIResponse.success_response(
        CreateChatbotSessionResponse(
            session_id=chatbot_session.id, chatbot_ref=chatbot_session.chatbot_ref
        )
    )


@router.get(
    "/sessions",
    description=load_description("chatbot/get_latest_session.md"),
    response_model=APIResponse[ChatbotSessionSummary],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def read_latest_chatbot_session(
    contact_ref: Annotated[
        str,
        Query(
            min_length=1,
            max_length=CONTACT_REF_MAX_LENGTH,
            description="Caller-side identifier of the end user whose latest session to fetch",
        ),
    ],
    session: SessionDep,
    auth_context: AuthContextDep,
    chatbot_ref: Annotated[
        str | None,
        Query(
            min_length=1,
            max_length=CHATBOT_REF_MAX_LENGTH,
            description="Chatbot whose latest session to fetch; omit to get the contact's newest session across all chatbots",
        ),
    ] = None,
) -> APIResponse[ChatbotSessionSummary]:
    chatbot_session = get_latest_chatbot_session(
        session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        contact_ref=contact_ref,
        chatbot_ref=chatbot_ref,
    )
    if chatbot_session is None:
        raise HTTPException(
            status_code=404,
            detail="No chatbot session found for this contact_ref"
            + (" and chatbot_ref" if chatbot_ref is not None else ""),
        )
    return APIResponse.success_response(
        ChatbotSessionSummary(
            session_id=chatbot_session.id,
            name=chatbot_session.name,
            contact_ref=chatbot_session.contact_ref,
            chatbot_ref=chatbot_session.chatbot_ref,
            system_prompt=chatbot_session.system_prompt,
            status=chatbot_session.status,
            completion_status=chatbot_session.completion_status,
            collected_fields=chatbot_session.collected_fields,
            config_id=chatbot_session.config_id,
            config_version=chatbot_session.config_version,
            inserted_at=chatbot_session.inserted_at,
            updated_at=chatbot_session.updated_at,
        )
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
    if chatbot_session.status == ChatbotSessionStatusEnum.INACTIVE:
        raise HTTPException(
            status_code=409,
            detail="Chatbot session is inactive; start a new session",
        )

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
