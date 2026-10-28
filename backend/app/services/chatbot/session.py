import asyncio
from uuid import UUID

from fastapi import HTTPException
from openai import OpenAI
from sqlmodel import Session

from app.core.db import engine
from app.crud.chatbot import (
    create_chatbot_session_message,
    get_chatbot_session,
    list_chatbot_session_messages,
    update_chatbot_session_progress,
)
from app.models import (
    ChatbotAgentTypeEnum,
    ChatbotMessageRoleEnum,
    ChatbotSessionMessage,
    ChatbotSessionStatusEnum,
)
from app.services.chatbot.agents.registry import run_agent_turn
from app.services.chatbot.config import (
    prepare_rag_search_client,
    resolve_chatbot_model_settings,
)
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.types import ChatbotMessageDict
from app.services.chatbot.utils import ChatbotModelSettings
from app.utils import APIResponse, get_webhook_secret, send_callback

SUPERSEDED_SESSION_ERROR = (
    "[KAAPI] Chatbot session was superseded by a newer session; start a new session."
)


def _split_pending_turn(
    stored_messages: list[ChatbotSessionMessage],
) -> tuple[list[ChatbotMessageDict], str | None]:
    """Split the stored transcript into (prior history, pending user message)."""
    history: list[ChatbotMessageDict] = []
    for stored in stored_messages:
        # The system prompt lives on the session row and is passed separately.
        if stored.role == ChatbotMessageRoleEnum.SYSTEM:
            continue
        history.append({"role": stored.role.value, "content": stored.content})

    if not history:
        return history, None
    if history[-1]["role"] != ChatbotMessageRoleEnum.USER:
        # No pending user turn; the agent runner rejects this with ChatbotInputError.
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
        # The turn may have been queued before a newer session superseded this one.
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

        system_prompt = chatbot_session.system_prompt
        chatbot_ref = chatbot_session.chatbot_ref
        session_status = chatbot_session.status.value
        agent_type = chatbot_session.agent_type
        collected_fields = dict(chatbot_session.collected_fields)
        openai_client: OpenAI | None = None

        try:
            model_settings: ChatbotModelSettings = resolve_chatbot_model_settings(
                session=session,
                project_id=project_id,
                config_id=chatbot_session.config_id,
                config_version=chatbot_session.config_version,
                max_tokens=None,
                history_token_limit=None,
            )
            if agent_type == ChatbotAgentTypeEnum.RAG:
                openai_client = prepare_rag_search_client(
                    session=session,
                    organization_id=organization_id,
                    project_id=project_id,
                    model_settings=model_settings,
                )
        except HTTPException as e:
            _send_failure_webhook(
                session_id=session_id,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
                error=str(e.detail),
            )
            return
        except ChatbotInputError as e:
            _send_failure_webhook(
                session_id=session_id,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
                error=str(e),
            )
            return

    history, user_message = _split_pending_turn(stored_messages)

    try:
        turn_result = asyncio.run(
            run_agent_turn(
                agent_type=agent_type,
                system_prompt=system_prompt,
                history=history,
                user_message=user_message,
                model_settings=model_settings,
                collected_fields=collected_fields,
                openai_client=openai_client,
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
            "chatbot_ref": chatbot_ref,
            "agent_type": agent_type.value,
            "reply": reply,
            "status": session_status,
            "completion_status": completion_status.value,
            "collected_fields": turn_result.collected_fields,
            "sources": turn_result.sources,
        }
    )
    send_callback(webhook_url, envelope.model_dump(exclude_none=True), webhook_secret)
    return
