from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import HttpUrl, model_validator
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel

from app.core.util import now
from app.models.organization import Organization
from app.models.project import Project


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
    collection_blob: dict[str, Any] | None = Field(
        sa_column=sa.Column(
            JSONB,
            nullable=True,
            comment="Provider-specific collection parameters (name, description, chunking params etc.)",
        )
    )
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


class DocumentInput(SQLModel):
    """Document to be added to knowledge base."""

    name: str | None = Field(
        description="Display name for the document",
    )
    id: UUID = Field(
        description="Reference to uploaded file/document in Kaapi",
    )


class CreateCollectionParams(SQLModel):
    """Request-specific parameters for knowledge base creation."""

    name: str | None = Field(
        min_length=1,
        description="Name of the knowledge base to create or update",
    )
    description: str | None = Field(
        default=None,
        description="Description of the knowledge base (required by Bedrock, optional for others)",
    )
    documents: list[DocumentInput] = Field(
        default_factory=list,
        description="List of documents to add to the knowledge base",
    )
    chunking_params: dict[str, Any] | None = Field(
        default=None,
        description="Chunking parameters for document processing (e.g., chunk_size, chunk_overlap)",
    )
    additional_params: dict[str, Any] | None = Field(
        default=None,
        description="Additional provider-specific parameters",
    )

    def model_post_init(self, __context: Any):
        """Deduplicate documents by file_id."""
        seen = set()
        unique_docs = []
        for doc in self.documents:
            if doc.file_id not in seen:
                seen.add(doc.file_id)
                unique_docs.append(doc)
        self.documents = unique_docs


class AssistantOptions(SQLModel):
    # Fields to be passed along to OpenAI. They must be a subset of
    # parameters accepted by the OpenAI.client.beta.assistants.create
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
    """Optional callback configuration for async job notifications."""

    callback_url: HttpUrl | None = Field(
        default=None,
        description="URL to call to report endpoint status",
    )


class ProviderOptions(SQLModel):
    """LLM provider configuration."""

    provider: Literal["openai"] = Field(
        default="openai", description="LLM provider to use for this collection"
    )


class CreationRequest(AssistantOptions, ProviderOptions, CallbackRequest):
    """API request for collection creation"""

    collection_params: CreateCollectionParams = Field(
        ...,
        description="Collection creation specific parameters (name, documents, etc.)",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Number of documents to process in a single batch",
    )


class DeletionRequest(ProviderOptions, CallbackRequest):

    """API request for collection deletion"""

    collection_id: UUID = Field(description="Collection to delete")
