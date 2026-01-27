from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import HttpUrl, model_validator
from sqlalchemy import UniqueConstraint, Index, text
from sqlmodel import Field, Relationship, SQLModel

from app.core.util import now
from app.models.document import DocumentPublic
from .project import Project


class ProviderType(str, Enum):
    """Supported LLM providers for collections."""

    openai = "openai"
    # BEDROCK = "bedrock"
    # GEMINI = "gemini"


class Collection(SQLModel, table=True):
    """Database model for Collection operations."""

    __table_args__ = (
        Index(
            "uq_collection_project_id_name_active",
            "project_id",
            "name",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="Unique identifier for the collection",
        sa_column_kwargs={"comment": "Unique identifier for the collection"},
    )
    provider: ProviderType = (
        Field(
            nullable=False,
            description="LLM provider used for this collection (e.g., 'openai', 'bedrock', 'gemini', etc)",
            sa_column_kwargs={"comment": "LLM provider used for this collection"},
        ),
    )
    llm_service_id: str = Field(
        nullable=False,
        description="External LLM service identifier (e.g., OpenAI vector store ID)",
        sa_column_kwargs={
            "comment": "External LLM service identifier (e.g., OpenAI vector store ID)"
        },
    )
    llm_service_name: str = Field(
        nullable=False,
        description="Name of the LLM service",
        sa_column_kwargs={"comment": "Name of the LLM service"},
    )
    name: str | None = Field(
        nullable=True,
        description="Name of the collection",
        sa_column_kwargs={"comment": "Name of the collection"},
    )
    description: str | None = Field(
        nullable=True,
        description="Description of the collection",
        sa_column_kwargs={"comment": "Description of the collection"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        description="Project the collection belongs to",
        sa_column_kwargs={"comment": "Reference to the project"},
    )
    inserted_at: datetime = Field(
        default_factory=now,
        description="Timestamp when the collection was created",
        sa_column_kwargs={"comment": "Timestamp when the collection was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        description="Timestamp when the collection was updated",
        sa_column_kwargs={"comment": "Timestamp when the collection was last updated"},
    )
    deleted_at: datetime | None = Field(
        default=None,
        description="Timestamp when the collection was deleted",
        sa_column_kwargs={"comment": "Timestamp when the collection was deleted"},
    )
    project: Project = Relationship(back_populates="collections")


# Request models
class CollectionOptions(SQLModel):
    name: str | None = Field(default=None, description="Name of the collection")
    description: str | None = Field(
        default=None, description="Description of the collection"
    )
    documents: list[UUID] = Field(
        description="List of document IDs",
    )
    batch_size: int = Field(
        default=1,
        description=(
            "Number of documents to send to OpenAI in a single "
            "transaction. See the `file_ids` parameter in the "
            "vector store [create batch](https://platform.openai.com/docs/api-reference/vector-stores-file-batches/createBatch)."
        ),
    )

    def model_post_init(self, __context: Any):
        self.documents = list(set(self.documents))


class AssistantOptions(SQLModel):
    # Fields to be passed along to OpenAI. They must be a subset of
    # parameters accepted by the OpenAI.clien.beta.assistants.create
    # API.
    model: str | None = Field(
        default=None,
        description=(
            "**[Deprecated]**  "
            "OpenAI model to attach to this assistant. The model "
            "must be compatable with the assistants API; see the "
            "OpenAI [model documentation](https://platform.openai.com/docs/models/compare) for more."
        ),
    )

    instructions: str | None = Field(
        default=None,
        description=(
            "**[Deprecated]**  "
            "Assistant instruction. Sometimes referred to as the "
            '"system" prompt.'
        ),
    )
    temperature: float = Field(
        default=1e-6,
        description=(
            "**[Deprecated]**  "
            "Model temperature. The default is slightly "
            "greater-than zero because it is [unknown how OpenAI "
            "handles zero](https://community.openai.com/t/clarifications-on-setting-temperature-0/886447/5)."
        ),
    )

    @model_validator(mode="before")
    def _assistant_fields_all_or_none(cls, values: dict[str, Any]) -> dict[str, Any]:
        def norm(x: Any) -> Any:
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip()
                return s if s else None
            return x  # let Pydantic handle non-strings

        model = norm(values.get("model"))
        instructions = norm(values.get("instructions"))

        if (model is None) ^ (instructions is None):
            raise ValueError(
                "To create an Assistant, provide BOTH 'model' and 'instructions'. "
                "If you only want a vector store, remove both fields."
            )

        values["model"] = model
        values["instructions"] = instructions
        return values


class CallbackRequest(SQLModel):
    callback_url: HttpUrl | None = Field(
        default=None,
        description="URL to call to report endpoint status",
    )


class ProviderOptions(SQLModel):
    """LLM provider configuration."""

    provider: Literal["openai"] = Field(
        default="openai", description="LLM provider to use for this collection"
    )


class CreationRequest(
    AssistantOptions,
    CollectionOptions,
    ProviderOptions,
    CallbackRequest,
):
    def extract_super_type(self, cls: "CreationRequest"):
        for field_name in cls.model_fields.keys():
            field_value = getattr(self, field_name)
            yield (field_name, field_value)


class DeletionRequest(CallbackRequest):
    collection_id: UUID = Field(description="Collection to delete")


# Response models


class CollectionIDPublic(SQLModel):
    id: UUID


class CollectionPublic(SQLModel):
    id: UUID
    llm_service_id: str
    llm_service_name: str
    project_id: int

    inserted_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class CollectionWithDocsPublic(CollectionPublic):
    documents: list[DocumentPublic] | None = None
