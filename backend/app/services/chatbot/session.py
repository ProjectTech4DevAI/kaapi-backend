import asyncio
from typing import NamedTuple
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from app.core.db import engine
from app.crud.chatbot import (
    create_chatbot_session,
    create_chatbot_session_message,
    get_chatbot_flow_version_by_id,
    get_chatbot_session,
    get_latest_chatbot_session,
    get_live_chatbot_flow_version,
    list_chatbot_session_messages,
    update_chatbot_session_progress,
)
from app.models import (
    AGENT_NODE_TYPES,
    ChatbotFlowNodeTypeEnum,
    ChatbotFlowVersion,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionMessage,
    ChatbotSessionStatusEnum,
    ChatbotSessionSummary,
)
from app.services.chatbot.agents.onboarding.graph import run_chatbot_turn
from app.services.chatbot.config import resolve_chatbot_model_settings
from app.services.chatbot.exceptions import (
    ChatbotFlowConflictError,
    ChatbotInputError,
    ChatbotProviderError,
)
from app.services.chatbot.flow import (
    find_flow_node,
    get_chatbot_or_raise,
    parse_flow_edges,
    parse_flow_nodes,
    resolve_entry_node,
)
from app.services.chatbot.types import ChatbotMessageDict
from app.services.chatbot.utils import ChatbotModelSettings
from app.utils import APIResponse, get_webhook_secret, send_callback

SUPERSEDED_SESSION_ERROR = (
    "[KAAPI] Chatbot session was superseded by a newer session; start a new session."
)
NO_LIVE_FLOW_MESSAGE = "[KAAPI] Chatbot has no published flow; publish the flow first."


class _TurnNode(NamedTuple):
    node_id: str
    prompt: str
    config_id: UUID
    config_version: int | None


def start_chatbot_session(
    *,
    session: Session,
    chatbot_id: UUID,
    organization_id: int,
    project_id: int,
    contact_ref: str,
) -> tuple[ChatbotSession, ChatbotFlowVersion]:
    """Create a session pinned to the live flow version and its entry node."""
    get_chatbot_or_raise(
        session=session,
        chatbot_id=chatbot_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    live_version = get_live_chatbot_flow_version(session, chatbot_id=chatbot_id)
    if live_version is None:
        raise ChatbotFlowConflictError(NO_LIVE_FLOW_MESSAGE)

    entry_node = resolve_entry_node(
        nodes=parse_flow_nodes(live_version.nodes),
        edges=parse_flow_edges(live_version.edges),
    )
    chatbot_session = create_chatbot_session(
        session,
        chatbot_id=chatbot_id,
        flow_version_id=live_version.id,
        contact_ref=contact_ref,
        active_node_id=entry_node.id,
    )
    return chatbot_session, live_version


def get_latest_chatbot_session_summary(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    contact_ref: str,
    chatbot_id: UUID | None,
) -> ChatbotSessionSummary | None:
    row = get_latest_chatbot_session(
        session,
        organization_id=organization_id,
        project_id=project_id,
        contact_ref=contact_ref,
        chatbot_id=chatbot_id,
    )
    if row is None:
        return None
    chatbot_session, chatbot, flow_version = row

    prompt: str | None = None
    active_node = find_flow_node(
        parse_flow_nodes(flow_version.nodes), chatbot_session.active_node_id
    )
    if active_node is not None and active_node.type in AGENT_NODE_TYPES:
        prompt = active_node.agent_data().prompt

    return ChatbotSessionSummary(
        session_id=chatbot_session.id,
        chatbot_id=chatbot.id,
        chatbot_name=chatbot.name,
        flow_version=flow_version.version,
        active_node_id=chatbot_session.active_node_id,
        prompt=prompt,
        contact_ref=chatbot_session.contact_ref,
        status=chatbot_session.status,
        completion_status=chatbot_session.completion_status,
        collected_fields=chatbot_session.collected_fields,
        inserted_at=chatbot_session.inserted_at,
        updated_at=chatbot_session.updated_at,
    )


def _resolve_turn_node(
    *, flow_version: ChatbotFlowVersion | None, active_node_id: str
) -> _TurnNode:
    """The runnable onboarding node the session is on; ChatbotInputError when it can't run."""
    active_node = None
    if flow_version is not None:
        active_node = find_flow_node(
            parse_flow_nodes(flow_version.nodes), active_node_id
        )
    if active_node is None:
        error_message = (
            f"[KAAPI] Active node '{active_node_id}' is not in the session's flow "
            "version; start a new session."
        )
        raise ChatbotInputError(error_message)

    if active_node.type != ChatbotFlowNodeTypeEnum.ONBOARDING:
        error_message = (
            f"[KAAPI] Active node '{active_node_id}' has type '{active_node.type.value}', "
            f"which the chatbot runtime does not support yet; only "
            f"'{ChatbotFlowNodeTypeEnum.ONBOARDING.value}' nodes can run."
        )
        raise ChatbotInputError(error_message)

    node_data = active_node.agent_data()
    if node_data.prompt is None or node_data.config_id is None:
        error_message = (
            f"[KAAPI] Active node '{active_node_id}' is missing data.prompt or "
            "data.config_id; republish the flow and start a new session."
        )
        raise ChatbotInputError(error_message)

    return _TurnNode(
        node_id=active_node.id,
        prompt=node_data.prompt,
        config_id=node_data.config_id,
        config_version=node_data.config_version,
    )


def _split_pending_turn(
    stored_messages: list[ChatbotSessionMessage],
) -> tuple[list[ChatbotMessageDict], str | None]:
    """Split the stored transcript into (prior history, pending user message)."""
    history: list[ChatbotMessageDict] = []
    for stored in stored_messages:
        # The system prompt lives on the flow node and is passed separately.
        if stored.role == ChatbotMessageRoleEnum.SYSTEM:
            continue
        history.append({"role": stored.role.value, "content": stored.content})

    if not history:
        return history, None
    if history[-1]["role"] != ChatbotMessageRoleEnum.USER:
        # No pending user turn; run_chatbot_turn rejects this with ChatbotInputError.
        return history, None
    pending = history.pop()
    return history, pending["content"]


def _send_failure_webhook(
    *,
    session_id: UUID,
    webhook_url: str,
    webhook_secret: str | None,
    error: str,
) -> None:
    envelope = APIResponse.failure_response(
        error=error, data={"session_id": str(session_id)}
    )
    send_callback(webhook_url, envelope.model_dump(exclude_none=True), webhook_secret)
    return


def process_chatbot_session_message(
    *,
    session_id: UUID,
    organization_id: int,
    project_id: int,
    webhook_url: str,
) -> None:
    """Run the pending chatbot turn for a session and POST the reply to `webhook_url`."""
    webhook_secret = get_webhook_secret(project_id, organization_id)

    # The DB session is closed before the LLM call so a slow provider doesn't
    # pin a pooled connection for the whole turn.
    with Session(engine) as session:
        chatbot_session = get_chatbot_session(
            session,
            session_id=session_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if chatbot_session is None:
            return
        # The turn may have been queued before a newer session (or a chatbot delete) deactivated this one.
        if chatbot_session.status == ChatbotSessionStatusEnum.INACTIVE:
            _send_failure_webhook(
                session_id=session_id,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
                error=SUPERSEDED_SESSION_ERROR,
            )
            return

        stored_messages = list_chatbot_session_messages(session, session_id=session_id)
        if (
            stored_messages
            and stored_messages[-1].role == ChatbotMessageRoleEnum.ASSISTANT
        ):
            return

        chatbot_id = chatbot_session.chatbot_id
        session_status = chatbot_session.status.value
        collected_fields = dict(chatbot_session.collected_fields)
        flow_version = get_chatbot_flow_version_by_id(
            session, flow_version_id=chatbot_session.flow_version_id
        )

        try:
            turn_node = _resolve_turn_node(
                flow_version=flow_version,
                active_node_id=chatbot_session.active_node_id,
            )
            model_settings: ChatbotModelSettings = resolve_chatbot_model_settings(
                session=session,
                organization_id=organization_id,
                project_id=project_id,
                config_id=turn_node.config_id,
                config_version=turn_node.config_version,
                max_tokens=None,
                history_token_limit=None,
            )
        except (HTTPException, ChatbotInputError) as e:
            error = str(e.detail) if isinstance(e, HTTPException) else str(e)
            _send_failure_webhook(
                session_id=session_id,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
                error=error,
            )
            return

    history, user_message = _split_pending_turn(stored_messages)

    try:
        turn_result = asyncio.run(
            run_chatbot_turn(
                system_prompt=turn_node.prompt,
                history=history,
                user_message=user_message,
                model_settings=model_settings,
                collected_fields=collected_fields,
            )
        )
    except (ChatbotInputError, ChatbotProviderError) as e:
        _send_failure_webhook(
            session_id=session_id,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            error=str(e),
        )
        return

    reply = turn_result.reply
    completion_status = turn_result.completion_status
    with Session(engine) as session:
        create_chatbot_session_message(
            session,
            session_id=session_id,
            role=ChatbotMessageRoleEnum.ASSISTANT,
            content=reply,
            node_id=turn_node.node_id,
        )
        update_chatbot_session_progress(
            session,
            session_id=session_id,
            collected_fields=turn_result.collected_fields,
            completion_status=completion_status,
        )

    envelope = APIResponse.success_response(
        data={
            "session_id": str(session_id),
            "chatbot_id": str(chatbot_id),
            "node_id": turn_node.node_id,
            "reply": reply,
            "status": session_status,
            "completion_status": completion_status.value,
            "collected_fields": turn_result.collected_fields,
        }
    )
    send_callback(webhook_url, envelope.model_dump(exclude_none=True), webhook_secret)
    return
