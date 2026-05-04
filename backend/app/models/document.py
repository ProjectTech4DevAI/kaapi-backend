from datetime import datetime
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlmodel import Field, Index, SQLModel, text

from app.core.util import now
from app.models.config.config import ConfigTag
from app.models.doc_transformation_job import TransformationStatus

_DOCUMENT_TAG_PG_ENUM = postgresql.ENUM(
    ConfigTag,
    name="config_tag",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
    create_type=False,
)


class DocumentBase(SQLModel):
    """Base model for documents with common fields."""

    fname: str = Field(
        description="The original filename of the document",
        sa_column_kwargs={"comment": "Original filename of the document"},
    )

    # Foreign keys
    project_id: int = Field(
        description="The ID of the project to which the document belongs",
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )


class Document(DocumentBase, table=True):
    """Database model for documents."""

    __table_args__ = (
        Index(
            "idx_document_project_id_tag_active",
            "project_id",
            "tag",
            text("inserted_at DESC"),
            postgresql_where=text("is_deleted IS FALSE"),
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="The unique identifier of the document",
        sa_column_kwargs={"comment": "Unique identifier for the document"},
    )
    object_store_url: str = Field(
        sa_column_kwargs={"comment": "Cloud storage URL for the document"},
    )
    is_deleted: bool = Field(
        default=False,
        sa_column_kwargs={"comment": "Soft delete flag"},
    )
    file_size_kb: float | None = Field(
        default=None,
        description="The size of the document in kilobytes",
        sa_column_kwargs={"comment": "Size of the document in kilobytes (KB)"},
    )
    tag: ConfigTag = Field(
        default=ConfigTag.DEFAULT,
        sa_column=sa.Column(
            _DOCUMENT_TAG_PG_ENUM,
            nullable=False,
            server_default=sa.text("'default'::config_tag"),
            comment=(
                "Tag classifying the document: 'default' for general use, "
                "'ASSESSMENT' for assessment documents."
            ),
        ),
    )

    # Foreign keys
    source_document_id: UUID | None = Field(
        default=None,
        foreign_key="document.id",
        nullable=True,
        sa_column_kwargs={
            "comment": "Reference to source document if this is a transformation"
        },
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        description="The timestamp when the document was inserted",
        sa_column_kwargs={"comment": "Timestamp when the document was uploaded"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        description="The timestamp when the document was last updated",
        sa_column_kwargs={"comment": "Timestamp when the document was last updated"},
    )
    deleted_at: datetime | None = Field(
        default=None,
        sa_column_kwargs={"comment": "Timestamp when the document was deleted"},
    )


class DocumentPublic(DocumentBase):
    id: UUID = Field(description="The unique identifier of the document")
    signed_url: str | None = Field(
        default=None, description="A signed URL for accessing the document"
    )
    inserted_at: datetime = Field(
        description="The timestamp when the document was inserted"
    )
    updated_at: datetime = Field(
        description="The timestamp when the document was last updated"
    )


class TransformedDocumentPublic(DocumentPublic):
    source_document_id: UUID | None = Field(
        default=None,
        description="The ID of the source document if this document is a transformation",
    )


class TransformationJobInfo(SQLModel):
    message: str
    job_id: UUID = Field(description="The unique identifier of the transformation job")
    status: TransformationStatus
    transformer: str = Field(description="The name of the transformer used")
    status_check_url: str = Field(
        description="The URL to check the status of the transformation job"
    )


class DocumentUploadResponse(DocumentPublic):
    signed_url: str = Field(description="A signed URL for accessing the document")
    transformation_job: TransformationJobInfo | None = None


class DocTransformationJobPublic(SQLModel):
    job_id: UUID
    source_document_id: UUID
    status: TransformationStatus
    transformed_document: TransformedDocumentPublic | None = None
    error_message: str | None = None


class DocTransformationJobsPublic(SQLModel):
    jobs: list[DocTransformationJobPublic]
    jobs_not_found: list[UUID]
