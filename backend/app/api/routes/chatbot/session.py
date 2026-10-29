from typing import Annotated
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.celery.utils import start_chatbot_session_message
from app.crud.chatbot import (
    create_chatbot_session_message,
    get_chatbot_session,
    list_chatbot_session_messages,
)
from app.models import (
    CONTACT_REF_MAX_LENGTH,
    ChatbotMessageRoleEnum,
    ChatbotSessionStatusEnum,
    ChatbotSessionSummary,
    CreateChatbotSessionRequest,
    CreateChatbotSessionResponse,
    SendChatbotSessionMessageRequest,
    SendChatbotSessionMessageResponse,
)
from app.services.chatbot.exceptions import (
    ChatbotFlowConflictError,
    ChatbotFlowValidationError,
    ChatbotNotFoundError,
    ChatbotSessionConflictError,
)
from app.services.chatbot.session import (
    get_latest_chatbot_session_summary,
    start_chatbot_session,
)
from app.utils import APIResponse, load_description, validate_callback_url

DEFAULT_TRACE_ID = "N/A"

router = APIRouter()


@router.get(
    "/sessions/latest",
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
    chatbot_id: Annotated[
        UUID | None,
        Query(
            description="Chatbot whose latest session to fetch; omit to get the contact's newest session across the project's chatbots",
        ),
    ] = None,
) -> APIResponse[ChatbotSessionSummary]:
    summary = get_latest_chatbot_session_summary(
        session=session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        contact_ref=contact_ref,
        chatbot_id=chatbot_id,
    )
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="No chatbot session found for this contact_ref"
            + (" and chatbot_id" if chatbot_id is not None else ""),
        )
    return APIResponse.success_response(summary)


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
            node_id=chatbot_session.active_node_id,
        )

    start_chatbot_session_message(
        session_id=str(session_id),
        organization_id=organization_id,
        project_id=project_id,
        webhook_url=request.webhook_url,
        trace_id=correlation_id.get() or DEFAULT_TRACE_ID,
    )
    return APIResponse.success_response(SendChatbotSessionMessageResponse())


@router.post(
    "/{chatbot_id}/sessions",
    status_code=201,
    description=load_description("chatbot/create_session.md"),
    response_model=APIResponse[CreateChatbotSessionResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_chatbot_session_route(
    chatbot_id: Annotated[UUID, Path(description="Chatbot identifier")],
    request: CreateChatbotSessionRequest,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[CreateChatbotSessionResponse]:
    try:
        chatbot_session, flow_version = start_chatbot_session(
            session=session,
            chatbot_id=chatbot_id,
            organization_id=auth_context.organization_.id,
            project_id=auth_context.project_.id,
            contact_ref=request.contact_ref,
        )
    except ChatbotNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ChatbotFlowConflictError, ChatbotSessionConflictError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ChatbotFlowValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return APIResponse.success_response(
        CreateChatbotSessionResponse(
            session_id=chatbot_session.id,
            chatbot_id=chatbot_session.chatbot_id,
            flow_version=flow_version.version,
            active_node_id=chatbot_session.active_node_id,
        )
    )
