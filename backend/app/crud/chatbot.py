import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, select

from app.core.util import is_constraint_violation, now
from app.models import (
    ACTIVE_CHATBOT_SESSION_UNIQUE_INDEX,
    CHATBOT_FLOW_VERSION_UNIQUE_CONSTRAINT,
    LIVE_CHATBOT_FLOW_VERSION_UNIQUE_INDEX,
    Chatbot,
    ChatbotCompletionStatusEnum,
    ChatbotFlowVersion,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionMessage,
    ChatbotSessionStatusEnum,
    ChatbotUpdate,
)
from app.services.chatbot.exceptions import (
    CONCURRENT_FLOW_CONFLICT_MESSAGE,
    CONCURRENT_SESSION_CONFLICT_MESSAGE,
    ChatbotFlowConflictError,
    ChatbotSessionConflictError,
)

logger = logging.getLogger(__name__)


def create_chatbot(
    session: Session, *, organization_id: int, project_id: int, name: str
) -> Chatbot:
    timestamp = now()
    chatbot = Chatbot(
        organization_id=organization_id,
        project_id=project_id,
        name=name,
        inserted_at=timestamp,
        updated_at=timestamp,
    )
    session.add(chatbot)
    session.commit()
    session.refresh(chatbot)
    logger.info(
        f"[create_chatbot] Chatbot created | chatbot_id: {chatbot.id}, project_id: {project_id}"
    )
    return chatbot


def get_chatbot(
    session: Session,
    *,
    chatbot_id: UUID,
    organization_id: int,
    project_id: int,
    for_update: bool = False,
) -> Chatbot | None:
    """A non-deleted chatbot in the project; `for_update` row-locks it to serialize flow writes."""
    statement = select(Chatbot).where(
        Chatbot.id == chatbot_id,
        Chatbot.organization_id == organization_id,
        Chatbot.project_id == project_id,
        col(Chatbot.deleted_at).is_(None),
    )
    if for_update:
        statement = statement.with_for_update()
    return session.exec(statement).first()


def list_chatbots(
    session: Session,
    *,
    organization_id: int,
    project_id: int,
    skip: int,
    limit: int,
) -> list[Chatbot]:
    statement = (
        select(Chatbot)
        .where(
            Chatbot.organization_id == organization_id,
            Chatbot.project_id == project_id,
            col(Chatbot.deleted_at).is_(None),
        )
        .order_by(col(Chatbot.updated_at).desc())
        .offset(skip)
        .limit(limit)
    )
    return list(session.exec(statement).all())


def update_chatbot(
    session: Session, *, chatbot: Chatbot, chatbot_update: ChatbotUpdate
) -> Chatbot:
    update_data = chatbot_update.model_dump(exclude_unset=True, exclude_none=True)
    update_data["updated_at"] = now()
    chatbot.sqlmodel_update(update_data)
    session.add(chatbot)
    session.commit()
    session.refresh(chatbot)
    return chatbot


def delete_chatbot(session: Session, *, chatbot: Chatbot) -> Chatbot:
    """Soft-delete the chatbot and deactivate its active sessions in the same transaction."""
    timestamp = now()
    deactivated_count = _deactivate_active_chatbot_sessions(
        session, chatbot_id=chatbot.id, contact_ref=None, timestamp=timestamp
    )
    chatbot.sqlmodel_update({"deleted_at": timestamp, "updated_at": timestamp})
    session.add(chatbot)
    session.commit()
    session.refresh(chatbot)
    logger.info(
        f"[delete_chatbot] Chatbot soft-deleted | chatbot_id: {chatbot.id}, "
        f"deactivated_sessions: {deactivated_count}"
    )
    return chatbot


def get_latest_chatbot_flow_version(
    session: Session, *, chatbot_id: UUID
) -> ChatbotFlowVersion | None:
    """The highest version: the draft when it isn't live, otherwise the live one."""
    statement = (
        select(ChatbotFlowVersion)
        .where(ChatbotFlowVersion.chatbot_id == chatbot_id)
        .order_by(col(ChatbotFlowVersion.version).desc())
        .limit(1)
    )
    return session.exec(statement).first()


def get_live_chatbot_flow_version(
    session: Session, *, chatbot_id: UUID
) -> ChatbotFlowVersion | None:
    statement = select(ChatbotFlowVersion).where(
        ChatbotFlowVersion.chatbot_id == chatbot_id,
        col(ChatbotFlowVersion.is_live).is_(True),
    )
    return session.exec(statement).first()


def get_chatbot_flow_version(
    session: Session, *, chatbot_id: UUID, version: int
) -> ChatbotFlowVersion | None:
    statement = select(ChatbotFlowVersion).where(
        ChatbotFlowVersion.chatbot_id == chatbot_id,
        ChatbotFlowVersion.version == version,
    )
    return session.exec(statement).first()


def get_chatbot_flow_version_by_id(
    session: Session, *, flow_version_id: UUID
) -> ChatbotFlowVersion | None:
    return session.get(ChatbotFlowVersion, flow_version_id)


def create_chatbot_flow_version(
    session: Session,
    *,
    chatbot_id: UUID,
    version: int,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    is_live: bool = False,
) -> ChatbotFlowVersion:
    timestamp = now()
    flow_version = ChatbotFlowVersion(
        chatbot_id=chatbot_id,
        version=version,
        nodes=nodes,
        edges=edges,
        is_live=is_live,
        inserted_at=timestamp,
        updated_at=timestamp,
    )
    session.add(flow_version)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        if not is_constraint_violation(e, CHATBOT_FLOW_VERSION_UNIQUE_CONSTRAINT):
            raise
        raise ChatbotFlowConflictError(CONCURRENT_FLOW_CONFLICT_MESSAGE) from e
    session.refresh(flow_version)
    logger.info(
        f"[create_chatbot_flow_version] Flow version created | chatbot_id: {chatbot_id}, "
        f"version: {version}, is_live: {is_live}"
    )
    return flow_version


def update_chatbot_flow_version_graph(
    session: Session,
    *,
    flow_version: ChatbotFlowVersion,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> ChatbotFlowVersion:
    flow_version.sqlmodel_update({"nodes": nodes, "edges": edges, "updated_at": now()})
    session.add(flow_version)
    session.commit()
    session.refresh(flow_version)
    return flow_version


def publish_chatbot_flow_version(
    session: Session,
    *,
    flow_version: ChatbotFlowVersion,
    nodes: list[dict[str, Any]],
) -> ChatbotFlowVersion:
    """Make `flow_version` the live one (storing its pinned `nodes`) in one transaction."""
    timestamp = now()
    # Must hit the DB before the draft is marked live, or the partial unique
    # index on (chatbot_id) WHERE is_live sees two live rows.
    unset_previous_live = (
        update(ChatbotFlowVersion)
        .where(
            col(ChatbotFlowVersion.chatbot_id) == flow_version.chatbot_id,
            col(ChatbotFlowVersion.is_live).is_(True),
        )
        .values(is_live=False, updated_at=timestamp)
    )
    session.exec(unset_previous_live)

    flow_version.sqlmodel_update(
        {"nodes": nodes, "is_live": True, "updated_at": timestamp}
    )
    session.add(flow_version)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        if not is_constraint_violation(e, LIVE_CHATBOT_FLOW_VERSION_UNIQUE_INDEX):
            raise
        raise ChatbotFlowConflictError(CONCURRENT_FLOW_CONFLICT_MESSAGE) from e
    session.refresh(flow_version)
    logger.info(
        f"[publish_chatbot_flow_version] Flow published | chatbot_id: {flow_version.chatbot_id}, "
        f"version: {flow_version.version}"
    )
    return flow_version


def _deactivate_active_chatbot_sessions(
    session: Session,
    *,
    chatbot_id: UUID,
    contact_ref: str | None,
    timestamp: datetime,
) -> int:
    """Mark active sessions of the chatbot (one contact's, or all when `contact_ref` is None) inactive; does not commit."""
    scope: list[ColumnElement[bool]] = [
        col(ChatbotSession.chatbot_id) == chatbot_id,
        col(ChatbotSession.status) == ChatbotSessionStatusEnum.ACTIVE,
    ]
    if contact_ref is not None:
        scope.append(col(ChatbotSession.contact_ref) == contact_ref)

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
    chatbot_id: UUID,
    flow_version_id: UUID,
    contact_ref: str,
    active_node_id: str,
) -> ChatbotSession:
    """Insert a new active session, superseding the contact's active ones for the same chatbot."""
    timestamp = now()
    _deactivate_active_chatbot_sessions(
        session,
        chatbot_id=chatbot_id,
        contact_ref=contact_ref,
        timestamp=timestamp,
    )
    chatbot_session = ChatbotSession(
        chatbot_id=chatbot_id,
        flow_version_id=flow_version_id,
        contact_ref=contact_ref,
        active_node_id=active_node_id,
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
    logger.info(
        f"[create_chatbot_session] Session created | session_id: {chatbot_session.id}, "
        f"chatbot_id: {chatbot_id}"
    )
    return chatbot_session


def get_chatbot_session(
    session: Session,
    *,
    session_id: UUID,
    organization_id: int,
    project_id: int,
) -> ChatbotSession | None:
    """Scoped through the owning chatbot; a deleted chatbot's sessions still resolve (they are inactive)."""
    statement = (
        select(ChatbotSession)
        .join(Chatbot, col(Chatbot.id) == col(ChatbotSession.chatbot_id))
        .where(
            ChatbotSession.id == session_id,
            Chatbot.organization_id == organization_id,
            Chatbot.project_id == project_id,
        )
    )
    return session.exec(statement).first()


def get_latest_chatbot_session(
    session: Session,
    *,
    organization_id: int,
    project_id: int,
    contact_ref: str,
    chatbot_id: UUID | None = None,
) -> tuple[ChatbotSession, Chatbot, ChatbotFlowVersion] | None:
    """Newest session of the contact across the project's non-deleted chatbots, or one chatbot's."""
    statement = (
        select(ChatbotSession, Chatbot, ChatbotFlowVersion)
        .join(Chatbot, col(Chatbot.id) == col(ChatbotSession.chatbot_id))
        .join(
            ChatbotFlowVersion,
            col(ChatbotFlowVersion.id) == col(ChatbotSession.flow_version_id),
        )
        .where(
            ChatbotSession.contact_ref == contact_ref,
            Chatbot.organization_id == organization_id,
            Chatbot.project_id == project_id,
            col(Chatbot.deleted_at).is_(None),
        )
    )
    if chatbot_id is not None:
        statement = statement.where(ChatbotSession.chatbot_id == chatbot_id)
    statement = statement.order_by(col(ChatbotSession.inserted_at).desc()).limit(1)
    row = session.exec(statement).first()
    if row is None:
        return None
    chatbot_session, chatbot, flow_version = row
    return chatbot_session, chatbot, flow_version


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
    node_id: str | None,
) -> ChatbotSessionMessage:
    message = ChatbotSessionMessage(
        session_id=session_id,
        role=role,
        content=content,
        node_id=node_id,
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
