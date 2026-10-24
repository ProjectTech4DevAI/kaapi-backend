from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, select

from app.core.util import is_constraint_violation, now
from app.models import (
    ACTIVE_CHATBOT_SESSION_UNIQUE_INDEX,
    ChatbotCompletionStatusEnum,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionMessage,
    ChatbotSessionStatusEnum,
)
from app.services.chatbot.exceptions import (
    CONCURRENT_SESSION_CONFLICT_MESSAGE,
    ChatbotSessionConflictError,
)


def _deactivate_active_chatbot_sessions(
    session: Session,
    *,
    organization_id: int,
    project_id: int,
    contact_ref: str,
    chatbot_ref: str,
    timestamp: datetime,
) -> int:
    """Mark this contact's active sessions of this chatbot inactive; does not commit."""
    scope = (
        col(ChatbotSession.organization_id) == organization_id,
        col(ChatbotSession.project_id) == project_id,
        col(ChatbotSession.contact_ref) == contact_ref,
        col(ChatbotSession.chatbot_ref) == chatbot_ref,
        col(ChatbotSession.status) == ChatbotSessionStatusEnum.ACTIVE,
    )

    # Runs first: once the second statement flips status, these rows no longer match.
    abandon_in_progress = (
        update(ChatbotSession)
        .where(
            *scope,
            col(ChatbotSession.completion_status)
            == ChatbotCompletionStatusEnum.IN_PROGRESS,
        )
        .values(
            status=ChatbotSessionStatusEnum.INACTIVE,
            completion_status=ChatbotCompletionStatusEnum.ABANDONED,
            updated_at=timestamp,
        )
    )
    abandoned_count = session.exec(abandon_in_progress).rowcount

    deactivate_remaining = (
        update(ChatbotSession)
        .where(*scope)
        .values(status=ChatbotSessionStatusEnum.INACTIVE, updated_at=timestamp)
    )
    deactivated_count = session.exec(deactivate_remaining).rowcount

    return abandoned_count + deactivated_count


def create_chatbot_session(
    session: Session,
    *,
    organization_id: int,
    project_id: int,
    name: str,
    contact_ref: str,
    chatbot_ref: str,
    config_id: UUID,
    config_version: int | None,
    system_prompt: str,
) -> ChatbotSession:
    """Insert a new active session, superseding the contact's active ones for the same chatbot."""
    timestamp = now()
    _deactivate_active_chatbot_sessions(
        session,
        organization_id=organization_id,
        project_id=project_id,
        contact_ref=contact_ref,
        chatbot_ref=chatbot_ref,
        timestamp=timestamp,
    )
    chatbot_session = ChatbotSession(
        organization_id=organization_id,
        project_id=project_id,
        name=name,
        contact_ref=contact_ref,
        chatbot_ref=chatbot_ref,
        config_id=config_id,
        config_version=config_version,
        system_prompt=system_prompt,
        inserted_at=timestamp,
        updated_at=timestamp,
    )
    session.add(chatbot_session)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        if not is_constraint_violation(e, ACTIVE_CHATBOT_SESSION_UNIQUE_INDEX):
            raise
        raise ChatbotSessionConflictError(CONCURRENT_SESSION_CONFLICT_MESSAGE) from e
    session.refresh(chatbot_session)
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


def update_chatbot_session_progress(
    session: Session,
    *,
    session_id: UUID,
    collected_fields: dict[str, Any],
    completion_status: ChatbotCompletionStatusEnum,
) -> ChatbotSession | None:
    chatbot_session = session.get(ChatbotSession, session_id)
    if chatbot_session is None:
        return None

    chatbot_session.sqlmodel_update(
        {
            "collected_fields": collected_fields,
            "completion_status": completion_status,
            "updated_at": now(),
        }
    )
    session.add(chatbot_session)
    session.commit()
    session.refresh(chatbot_session)
    return chatbot_session


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
    return message


def list_chatbot_session_messages(
    session: Session, *, session_id: UUID
) -> list[ChatbotSessionMessage]:
    statement = (
        select(ChatbotSessionMessage)
        .where(ChatbotSessionMessage.session_id == session_id)
        .order_by(col(ChatbotSessionMessage.inserted_at).asc())
    )
    return list(session.exec(statement).all())


def chatbot_ref_exists(session: Session, *, project_id: int, chatbot_ref: str) -> bool:
    statement = (
        select(ChatbotSession.id)
        .where(
            ChatbotSession.project_id == project_id,
            ChatbotSession.chatbot_ref == chatbot_ref,
        )
        .limit(1)
    )
    return session.exec(statement).first() is not None


def get_latest_chatbot_session(
    session: Session,
    *,
    organization_id: int,
    project_id: int,
    contact_ref: str,
    chatbot_ref: str | None = None,
) -> ChatbotSession | None:
    statement = select(ChatbotSession).where(
        ChatbotSession.organization_id == organization_id,
        ChatbotSession.project_id == project_id,
        ChatbotSession.contact_ref == contact_ref,
    )
    if chatbot_ref is not None:
        statement = statement.where(ChatbotSession.chatbot_ref == chatbot_ref)
    statement = statement.order_by(col(ChatbotSession.inserted_at).desc()).limit(1)
    return session.exec(statement).first()
