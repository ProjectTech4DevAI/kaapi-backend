from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import HttpUrl, model_validator
from sqlmodel import Field, Relationship, SQLModel

from app.core.util import now
from app.models.document import DocumentPublic

from .organization import Organization
from .project import Project


class Collection(SQLModel, table=True):
    """Database model for Collection operations."""

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the collection"},
    )
    llm_service_id: str = Field(
        nullable=False,
        sa_column_kwargs={
            "comment": "External LLM service identifier (e.g., OpenAI vector store ID)"
        },
    )
    llm_service_name: str = Field(
        nullable=False,
        sa_column_kwargs={"comment": "Name of the LLM service"},
    )

    # Foreign keys
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        sa_column_kwargs={"comment": "Timestamp when the collection was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        sa_column_kwargs={"comment": "Timestamp when the collection was last updated"},
    )
    deleted_at: datetime | None = Field(
        default=None,
        sa_column_kwargs={"comment": "Timestamp when the collection was deleted"},
    )

    # Relationships
    organization: Organization = Relationship(back_populates="collections")
    project: Project = Relationship(back_populates="collections")


# Request models
class DocumentOptions(SQLModel):
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
    DocumentOptions,
    ProviderOptions,
    AssistantOptions,
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
    organization_id: int

    inserted_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class CollectionWithDocsPublic(CollectionPublic):
    documents: list[DocumentPublic] | None = None
