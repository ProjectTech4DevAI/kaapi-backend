from datetime import datetime
from typing import Any
from uuid import UUID

from sqlmodel import SQLModel

from app.models.document import DocumentPublic


class CreateCollectionResult(SQLModel):
    llm_service_id: str
    llm_service_name: str
    collection_blob: dict[str, Any]


class CollectionIDPublic(SQLModel):
    id: UUID


class CollectionPublic(SQLModel):
    id: UUID
    llm_service_id: str
    llm_service_name: str
    project_id: int
    organization_id: int

    inserted_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class CollectionWithDocsPublic(CollectionPublic):
    documents: list[DocumentPublic] | None = None
