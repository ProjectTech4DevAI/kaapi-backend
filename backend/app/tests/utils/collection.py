from uuid import UUID, uuid4
from typing import Optional

from sqlmodel import Session

from app.models import (
    Collection,
    CollectionActionType,
    CollectionJob,
    CollectionJobStatus,
    ProviderType,
    Project,
)
from app.crud import CollectionCrud, CollectionJobCrud
from app.services.collections.helpers import get_service_name


class constants:
    openai_model = "gpt-4o"
    llm_service_name = "test-service-name"


def uuid_increment(value: UUID) -> UUID:
    inc = int(value) + 1
    return UUID(int=inc)


def get_assistant_collection(
    db: Session,
    project: Project,
    *,
    assistant_id: Optional[str] = None,
    model: str = "gpt-4o",
    collection_id: Optional[UUID] = None,
) -> Collection:
    """
    Create a Collection configured for the Assistant path.
    execute_job will treat this as `is_vector = False` and use assistant id.
    """
    if assistant_id is None:
        assistant_id = f"asst_{uuid4().hex}"

    collection = Collection(
        id=collection_id or uuid4(),
        project_id=project.id,
        organization_id=project.organization_id,
        llm_service_name=model,
        llm_service_id=assistant_id,
        provider=ProviderType.openai,
    )
    return CollectionCrud(db, project.id).create(collection)


def get_vector_store_collection(
    db: Session,
    project: Project,
    *,
    vector_store_id: Optional[str] = None,
    collection_id: Optional[UUID] = None,
) -> Collection:
    """
    Create a Collection configured for the Vector Store path.
    execute_job will treat this as `is_vector = True` and use vector store id.
    """
    if vector_store_id is None:
        vector_store_id = f"vs_{uuid4().hex}"

    collection = Collection(
        id=collection_id or uuid4(),
        project_id=project.id,
        llm_service_name=get_service_name("openai"),
        llm_service_id=vector_store_id,
        provider=ProviderType.openai,
    )
    return CollectionCrud(db, project.id).create(collection)


def get_collection_job(
    db: Session,
    project: Project,
    *,
    action_type: CollectionActionType = CollectionActionType.CREATE,
    status: CollectionJobStatus = CollectionJobStatus.PENDING,
    collection_id: Optional[UUID] = None,
    error_message: Optional[str] = None,
    job_id: Optional[UUID] = None,
) -> CollectionJob:
    """
    Generic seed for a CollectionJob row.
    """
    job = CollectionJob(
        id=job_id or uuid4(),
        project_id=project.id,
        action_type=action_type.value if hasattr(action_type, "value") else action_type,
        status=status.value if hasattr(status, "value") else status,
        error_message=error_message,
        collection_id=collection_id,
    )
    return CollectionJobCrud(db, project.id).create(job)
