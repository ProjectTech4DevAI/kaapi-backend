import logging
from typing import Optional
from sqlmodel import Session, select, func

from app.models import OpenAIConversation, OpenAIConversationCreate
from app.core.util import now

logger = logging.getLogger(__name__)


def get_conversation_by_id(
    session: Session, conversation_id: int, project_id: int
) -> OpenAIConversation | None:
    """
    Return a conversation by its ID and project.
    """
    statement = select(OpenAIConversation).where(
        OpenAIConversation.id == conversation_id,
        OpenAIConversation.project_id == project_id,
        OpenAIConversation.deleted_at.is_(None),
    )
    result = session.exec(statement).first()
    return result


def get_conversation_by_response_id(
    session: Session, response_id: str, project_id: int
) -> OpenAIConversation | None:
    """
    Return a conversation by its OpenAI response ID and project.
    """
    statement = select(OpenAIConversation).where(
        OpenAIConversation.response_id == response_id,
        OpenAIConversation.project_id == project_id,
        OpenAIConversation.deleted_at.is_(None),
    )
    result = session.exec(statement).first()
    return result


def get_conversation_by_ancestor_id(
    session: Session, ancestor_response_id: str, project_id: int
) -> OpenAIConversation | None:
    """
    Return the latest conversation by its ancestor response ID and project.
    """
    statement = (
        select(OpenAIConversation)
        .where(
            OpenAIConversation.ancestor_response_id == ancestor_response_id,
            OpenAIConversation.project_id == project_id,
            OpenAIConversation.deleted_at.is_(None),
        )
        .order_by(OpenAIConversation.inserted_at.desc())
        .limit(1)
    )
    result = session.exec(statement).first()
    return result


def get_ancestor_id_from_response(
    session: Session,
    current_response_id: str,
    previous_response_id: str | None,
    project_id: int,
) -> str:
    """
    Set the ancestor_response_id based on previous_response_id.

    Logic:
    1. If previous_response_id is None, then ancestor_response_id = current_response_id
    2. If previous_response_id is not None, look in db for that ID
       - If found, use that conversation's ancestor_id
       - If not found, ancestor_response_id = previous_response_id

    Args:
        session: Database session
        current_response_id: The current response ID
        previous_response_id: The previous response ID (can be None)
        project_id: The project ID for scoping the search

    Returns:
        str: The determined ancestor_response_id
    """
    if previous_response_id is None:
        # If previous_response_id is None, then ancestor_response_id = current_response_id
        return current_response_id

    # If previous_response_id is not None, look in db for that ID
    previous_conversation = get_conversation_by_response_id(
        session=session, response_id=previous_response_id, project_id=project_id
    )

    if previous_conversation:
        # If found, use that conversation's ancestor_id
        return previous_conversation.ancestor_response_id
    else:
        # If not found, ancestor_response_id = previous_response_id
        return previous_response_id


def get_conversations_count_by_project(
    session: Session,
    project_id: int,
) -> int:
    """
    Return the total count of conversations for a given project.
    """
    statement = select(func.count(OpenAIConversation.id)).where(
        OpenAIConversation.project_id == project_id,
        OpenAIConversation.deleted_at.is_(None),
    )
    result = session.exec(statement).one()
    return result


def get_conversations_by_project(
    session: Session,
    project_id: int,
    skip: int = 0,
    limit: int = 100,
) -> list[OpenAIConversation]:
    """
    Return all conversations for a given project, with optional pagination.
    """
    statement = (
        select(OpenAIConversation)
        .where(
            OpenAIConversation.project_id == project_id,
            OpenAIConversation.deleted_at.is_(None),
        )
        .order_by(OpenAIConversation.inserted_at.desc())
        .offset(skip)
        .limit(limit)
    )
    results = session.exec(statement).all()
    return results


def create_conversation(
    session: Session,
    conversation: OpenAIConversationCreate,
    project_id: int,
    organization_id: int,
) -> OpenAIConversation:
    """
    Create a new conversation in the database.
    """
    db_conversation = OpenAIConversation(
        **conversation.model_dump(),
        project_id=project_id,
        organization_id=organization_id,
    )
    session.add(db_conversation)
    session.commit()
    session.refresh(db_conversation)

    logger.info(
        f"Created conversation with response_id={db_conversation.response_id}, "
        f"assistant_id={db_conversation.assistant_id}, project_id={project_id}"
    )

    return db_conversation


def delete_conversation(
    session: Session,
    conversation_id: int,
    project_id: int,
) -> OpenAIConversation | None:
    """
    Soft delete a conversation by marking it as deleted.
    """
    db_conversation = get_conversation_by_id(session, conversation_id, project_id)
    if not db_conversation:
        return None

    db_conversation.deleted_at = now()
    session.add(db_conversation)
    session.commit()
    session.refresh(db_conversation)

    logger.info(
        f"Deleted conversation with id={conversation_id}, "
        f"response_id={db_conversation.response_id}, project_id={project_id}"
    )

    return db_conversation
