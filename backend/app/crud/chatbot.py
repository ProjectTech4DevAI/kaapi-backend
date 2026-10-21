import logging
from uuid import UUID

from sqlmodel import Session, col, select

from app.core.util import now
from app.models import (
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionMessage,
)

logger = logging.getLogger(__name__)


def create_chatbot_session(
    session: Session,
    *,
    organization_id: int,
    project_id: int,
    contact_ref: str,
    config_id: UUID,
    config_version: int | None,
    system_prompt: str,
) -> ChatbotSession:
    timestamp = now()
    chatbot_session = ChatbotSession(
        organization_id=organization_id,
        project_id=project_id,
        contact_ref=contact_ref,
        config_id=config_id,
        config_version=config_version,
        system_prompt=system_prompt,
        inserted_at=timestamp,
        updated_at=timestamp,
    )
    session.add(chatbot_session)
    session.commit()
    session.refresh(chatbot_session)
    logger.info(
        f"[create_chatbot_session] Chatbot session created | "
        f"session_id: {chatbot_session.id}, project_id: {project_id}, config_id: {config_id}"
    )
    return chatbot_session


def get_chatbot_session(
    session: Session,
    *,
    session_id: UUID,
    organization_id: int,
    project_id: int,
) -> ChatbotSession | None:
    statement = select(ChatbotSession).where(
        ChatbotSession.id == session_id,
        ChatbotSession.organization_id == organization_id,
        ChatbotSession.project_id == project_id,
    )
    return session.exec(statement).first()


def create_chatbot_session_message(
    session: Session,
    *,
    session_id: UUID,
    role: ChatbotMessageRoleEnum,
    content: str,
) -> ChatbotSessionMessage:
    message = ChatbotSessionMessage(
        session_id=session_id,
        role=role,
        content=content,
        inserted_at=now(),
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    logger.info(
        f"[create_chatbot_session_message] Chatbot message stored | "
        f"session_id: {session_id}, role: {role.value}, message_id: {message.id}"
    )
    return message


def list_chatbot_session_messages(
    session: Session, *, session_id: UUID
) -> list[ChatbotSessionMessage]:
    # Unpaginated on purpose: the engine needs the whole transcript each turn
    # and summarizes long histories itself.
    statement = (
        select(ChatbotSessionMessage)
        .where(ChatbotSessionMessage.session_id == session_id)
        .order_by(col(ChatbotSessionMessage.inserted_at).asc())
    )
    return list(session.exec(statement).all())


def list_chatbot_sessions_by_contact_ref(
    session: Session,
    *,
    contact_ref: str,
    organization_id: int,
    project_id: int,
) -> list[ChatbotSession]:
    statement = (
        select(ChatbotSession)
        .where(
            ChatbotSession.contact_ref == contact_ref,
            ChatbotSession.organization_id == organization_id,
            ChatbotSession.project_id == project_id,
        )
        .order_by(col(ChatbotSession.inserted_at).desc())
    )
    return list(session.exec(statement).all())
