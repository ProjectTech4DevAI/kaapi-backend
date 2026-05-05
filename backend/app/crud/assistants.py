import logging

from typing import Optional
import uuid

import openai
from fastapi import HTTPException
from openai import OpenAI
from openai.types.beta import Assistant as OpenAIAssistant
from sqlmodel import Session, and_, select

from app.models import Assistant, AssistantCreate, AssistantUpdate
from app.utils import mask_string
from app.core.util import now

logger = logging.getLogger(__name__)


def get_assistant_by_id(
    session: Session, assistant_id: str, project_id: int
) -> Optional[Assistant]:
    """Get an assistant by its OpenAI assistant ID and organization ID."""
    statement = select(Assistant).where(
        and_(
            Assistant.assistant_id == assistant_id,
            Assistant.project_id == project_id,
            Assistant.deleted_at.is_(None),
        )
    )
    return session.exec(statement).first()


def get_assistants_by_project(
    session: Session,
    project_id: int,
    skip: int = 0,
    limit: int = 100,
) -> list[Assistant]:
    """
    Return all assistants for a given project and organization, with optional pagination.
    """
    statement = (
        select(Assistant)
        .where(
            Assistant.project_id == project_id,
            Assistant.deleted_at.is_(None),
        )
        .offset(skip)
        .limit(limit)
    )
    results = session.exec(statement).all()
    return results


def fetch_assistant_from_openai(assistant_id: str, client: OpenAI) -> OpenAIAssistant:
    """
    Fetch an assistant from OpenAI.
    Returns OpenAI Assistant model.
    """

    try:
        assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
        return assistant
    except openai.NotFoundError as e:
        logger.error(
            f"[fetch_assistant_from_openai] Assistant not found: {mask_string(assistant_id)} | {e}"
        )
        raise HTTPException(status_code=404, detail="Assistant not found in OpenAI.")
    except openai.OpenAIError as e:
        logger.error(
            f"[fetch_assistant_from_openai] OpenAI API error while retrieving assistant {mask_string(assistant_id)}: {e}"
        )
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e}")


def verify_vector_store_ids_exist(
    openai_client: OpenAI, vector_store_ids: list[str]
) -> None:
    """
    Raises HTTPException if any of the vector_store_ids do not exist in OpenAI.
    """
    for vector_store_id in vector_store_ids:
        try:
            openai_client.vector_stores.retrieve(vector_store_id)
        except openai.NotFoundError:
            logger.error(f"Vector store ID {vector_store_id} not found in OpenAI.")
            raise HTTPException(
                status_code=400,
                detail=f"Vector store ID {vector_store_id} not found in OpenAI.",
            )
        except openai.OpenAIError as e:
            logger.error(f"Failed to verify vector store ID {vector_store_id}: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Error verifying vector store ID {vector_store_id}: {str(e)}",
            )


def sync_assistant(
    session: Session,
    organization_id: int,
    project_id: int,
    openai_assistant: OpenAIAssistant,
) -> Assistant:
    """
    Insert an assistant into the database by converting OpenAI Assistant to local Assistant model.
    """
    assistant_id = openai_assistant.id
    existing_assistant = get_assistant_by_id(session, assistant_id, project_id)
    if existing_assistant:
        logger.warning(
            f"[sync_assistant] Assistant with ID {mask_string(assistant_id)} already exists in the database. | project_id: {project_id}"
        )
        raise HTTPException(
            status_code=409,
            detail=f"Assistant with ID {assistant_id} already exists.",
        )

    if not openai_assistant.instructions:
        logger.warning(
            f"[sync_assistant] OpenAI assistant {mask_string(assistant_id)} has no instructions. | project_id: {project_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="Assistant has no instruction.",
        )

    vector_store_ids = []
    if openai_assistant.tool_resources and hasattr(
        openai_assistant.tool_resources, "file_search"
    ):
        file_search = openai_assistant.tool_resources.file_search
        if file_search and hasattr(file_search, "vector_store_ids"):
            vector_store_ids = file_search.vector_store_ids or []

    max_num_results = 20
    for tool in openai_assistant.tools or []:
        if tool.type == "file_search":
            file_search = getattr(tool, "file_search", None)
            if file_search and hasattr(file_search, "max_num_results"):
                max_num_results = file_search.max_num_results
            break

    db_assistant = Assistant(
        assistant_id=openai_assistant.id,
        name=openai_assistant.name or openai_assistant.id,
        instructions=openai_assistant.instructions,
        model=openai_assistant.model,
        vector_store_ids=vector_store_ids,
        temperature=openai_assistant.temperature
        if openai_assistant.temperature is not None
        else 0,
        max_num_results=max_num_results,
        project_id=project_id,
        organization_id=organization_id,
    )

    session.add(db_assistant)
    session.commit()
    session.refresh(db_assistant)

    logger.info(
        f"[sync_assistant] Successfully ingested assistant with ID {mask_string(assistant_id)}."
    )
    return db_assistant


def create_assistant(
    session: Session,
    openai_client: OpenAI,
    assistant: AssistantCreate,
    project_id: int,
    organization_id: int,
) -> Assistant:
    verify_vector_store_ids_exist(openai_client, assistant.vector_store_ids)

    assistant.assistant_id = assistant.assistant_id or str(uuid.uuid4())

    existing = get_assistant_by_id(session, assistant.assistant_id, project_id)
    if existing:
        logger.error(
            f"[create_assistant] Assistant with ID {mask_string(assistant.assistant_id)} already exists. | project_id: {project_id}"
        )
        raise HTTPException(
            status_code=409,
            detail=f"Assistant with ID {assistant.assistant_id} already exists.",
        )

    assistant = Assistant(
        **assistant.model_dump(exclude_unset=True),
        project_id=project_id,
        organization_id=organization_id,
    )
    session.add(assistant)
    session.commit()
    session.refresh(assistant)
    logger.info(
        f"[create_assistant] Assistant created successfully. | project_id: {project_id}, assistant_id: {mask_string(assistant.assistant_id)}"
    )
    return assistant


def update_assistant(
    session: Session,
    openai_client: OpenAI,
    assistant_id: str,
    project_id: int,
    assistant_update: AssistantUpdate,
) -> Assistant:
    existing_assistant = get_assistant_by_id(session, assistant_id, project_id)
    if not existing_assistant:
        logger.error(
            f"[update_assistant] Assistant {mask_string(assistant_id)} not found | project_id: {project_id}"
        )
        raise HTTPException(status_code=404, detail="Assistant not found.")

    # Update non-vector_store_ids fields, if present
    update_fields = assistant_update.model_dump(
        exclude_unset=True, exclude={"vector_store_ids_add", "vector_store_ids_remove"}
    )
    for field, value in update_fields.items():
        setattr(existing_assistant, field, value)

    current_vector_stores: set[str] = set(existing_assistant.vector_store_ids or [])

    # Validate for conflicting add/remove operations
    add_ids = set(assistant_update.vector_store_ids_add or [])
    remove_ids = set(assistant_update.vector_store_ids_remove or [])
    if conflicting_ids := add_ids & remove_ids:
        logger.error(
            f"[update_assistant] Conflicting vector store IDs in add/remove: {conflicting_ids} | project_id: {project_id}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Conflicting vector store IDs in add/remove: {conflicting_ids}.",
        )

    # Add new vector store IDs
    if add_ids:
        verify_vector_store_ids_exist(openai_client, list(add_ids))
        current_vector_stores.update(add_ids)

    # Remove vector store IDs
    if remove_ids:
        current_vector_stores.difference_update(remove_ids)

    # Update assistant's vector store IDs
    existing_assistant.vector_store_ids = list(current_vector_stores)
    existing_assistant.updated_at = now()
    session.add(existing_assistant)
    session.commit()
    session.refresh(existing_assistant)

    logger.info(
        f"[update_assistant] Assistant {mask_string(assistant_id)} updated successfully. | project_id: {project_id}"
    )
    return existing_assistant


def delete_assistant(
    session: Session,
    assistant_id: str,
    project_id: int,
) -> Assistant:
    """
    Soft delete an assistant by marking it as deleted.
    """
    existing_assistant = get_assistant_by_id(session, assistant_id, project_id)
    if not existing_assistant:
        logger.warning(
            f"[delete_assistant] Assistant {mask_string(assistant_id)} not found | project_id: {project_id}"
        )
        raise HTTPException(status_code=404, detail="Assistant not found.")

    existing_assistant.deleted_at = now()
    session.add(existing_assistant)
    session.commit()
    session.refresh(existing_assistant)

    logger.info(
        f"[delete_assistant] Assistant {mask_string(assistant_id)} soft deleted successfully. | project_id: {project_id}"
    )
    return existing_assistant
