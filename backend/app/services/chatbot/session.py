import asyncio
import logging
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from app.core.db import engine
from app.crud.chatbot import (
    create_chatbot_session_message,
    get_chatbot_session,
    list_chatbot_session_messages,
)
from app.models import ChatbotMessageRoleEnum, ChatbotSessionMessage
from app.services.chatbot.config import resolve_chatbot_model_settings
from app.services.chatbot.engine import run_chatbot_turn
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.services.chatbot.types import ChatbotMessageDict
from app.services.chatbot.utils import ChatbotModelSettings
from app.utils import get_webhook_secret, send_callback

logger = logging.getLogger(__name__)


def _split_pending_turn(
    stored_messages: list[ChatbotSessionMessage],
) -> tuple[list[ChatbotMessageDict], str | None]:
    """Split the stored transcript into (prior history, pending user message).

    The route persists the user's message before enqueueing, so it is the last
    stored row; the engine wants it separately from the history before it.
    """
    history: list[ChatbotMessageDict] = []
    for stored in stored_messages:
        # The system prompt lives on the session row and is passed separately.
        if stored.role == ChatbotMessageRoleEnum.SYSTEM:
            continue
        history.append({"role": stored.role.value, "content": stored.content})

    if not history:
        return history, None
    if history[-1]["role"] != ChatbotMessageRoleEnum.USER:
        # No pending user turn; the engine rejects this with ChatbotInputError.
        return history, None
    pending = history.pop()
    return history, pending["content"]


def _extract_reply(transcript: list[ChatbotMessageDict]) -> str:
    for entry in reversed(transcript):
        if entry["role"] == ChatbotMessageRoleEnum.ASSISTANT:
            return entry["content"]
    raise ChatbotInputError("[KAAPI] Chat model returned no assistant reply.")


def _send_failure_webhook(
    *,
    session_id: UUID,
    webhook_url: str,
    webhook_secret: str | None,
    error: str,
) -> None:
    payload: dict[str, Any] = {
        "success": False,
        "error": error,
        "data": {"session_id": str(session_id)},
    }
    delivered = send_callback(webhook_url, payload, webhook_secret)
    logger.info(
        f"[_send_failure_webhook] Failure webhook dispatched | "
        f"session_id: {session_id}, delivered: {delivered}"
    )


def process_chatbot_session_message(
    *,
    session_id: UUID,
    organization_id: int,
    project_id: int,
    webhook_url: str,
) -> None:
    """Run the pending chatbot turn for a session and POST the reply to `webhook_url`.

    Idempotent under Celery redelivery: if the transcript already ends on an
    assistant turn, the pending turn was answered and nothing is re-run.
    """
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
            logger.error(
                f"[process_chatbot_session_message] Chatbot session not found | "
                f"session_id: {session_id}, project_id: {project_id}"
            )
            return

        stored_messages = list_chatbot_session_messages(session, session_id=session_id)
        if (
            stored_messages
            and stored_messages[-1].role == ChatbotMessageRoleEnum.ASSISTANT
        ):
            logger.warning(
                f"[process_chatbot_session_message] Turn already answered, skipping | "
                f"session_id: {session_id}"
            )
            return

        system_prompt = chatbot_session.system_prompt
        session_status = chatbot_session.status.value
        collected_fields = dict(chatbot_session.collected_fields)

        try:
            model_settings: ChatbotModelSettings = resolve_chatbot_model_settings(
                session=session,
                project_id=project_id,
                config_id=chatbot_session.config_id,
                config_version=chatbot_session.config_version,
                max_tokens=None,
                history_token_limit=None,
            )
        except HTTPException as e:
            # The config was resolvable at session creation; it has since been
            # deleted or lost its version, and the caller only hears via webhook.
            logger.warning(
                f"[process_chatbot_session_message] Config no longer resolvable | "
                f"session_id: {session_id}, detail: {e.detail}"
            )
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
    logger.info(
        f"[process_chatbot_session_message] Running turn | session_id: {session_id}, "
        f"history_count: {len(history)}, has_user_message: {user_message is not None}"
    )

    try:
        transcript = asyncio.run(
            run_chatbot_turn(
                system_prompt=system_prompt,
                history=history,
                user_message=user_message,
                model_settings=model_settings,
            )
        )
        reply = _extract_reply(transcript)
    except (ChatbotInputError, ChatbotProviderError) as e:
        _send_failure_webhook(
            session_id=session_id,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            error=str(e),
        )
        return

    with Session(engine) as session:
        create_chatbot_session_message(
            session,
            session_id=session_id,
            role=ChatbotMessageRoleEnum.ASSISTANT,
            content=reply,
        )

    payload: dict[str, Any] = {
        "success": True,
        "data": {
            "session_id": str(session_id),
            "reply": reply,
            "status": session_status,
            "collected_fields": collected_fields,
        },
    }
    delivered = send_callback(webhook_url, payload, webhook_secret)
    logger.info(
        f"[process_chatbot_session_message] Turn completed | session_id: {session_id}, "
        f"reply_length: {len(reply)}, webhook_delivered: {delivered}"
    )
