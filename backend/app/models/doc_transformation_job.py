import enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import ConfigDict
from sqlmodel import Field, SQLModel

from app.core.util import now


class TransformationStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DocTransformationJob(SQLModel, table=True):
    """Database model for DocTransformationJob operations."""

    __tablename__ = "doc_transformation_job"  # pyright: ignore[reportAssignmentType]

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the transformation job"},
    )
    status: TransformationStatus = Field(
        default=TransformationStatus.PENDING,
        sa_column_kwargs={
            "comment": "Current status (PENDING, PROCESSING, COMPLETED, FAILED)"
        },
    )
    task_id: str | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Celery task ID for async processing"},
    )
    trace_id: str | None = Field(
        default=None,
        description="Tracing ID for correlating logs and traces.",
        sa_column_kwargs={"comment": "Tracing ID for correlating logs and traces"},
    )
    error_message: str | None = Field(
        default=None,
        sa_column_kwargs={"comment": "Error message if transformation failed"},
    )

    # Foreign keys
    source_document_id: UUID = Field(
        foreign_key="document.id",
        sa_column_kwargs={
            "comment": "Reference to the source document being transformed"
        },
    )
    transformed_document_id: UUID | None = Field(
        default=None,
        foreign_key="document.id",
        sa_column_kwargs={"comment": "Reference to the resulting transformed document"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        sa_column_kwargs={"comment": "Timestamp when the job was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        sa_column_kwargs={"comment": "Timestamp when the job was last updated"},
    )

    @property
    def job_id(self) -> UUID:
        return self.id

    @property
    def job_inserted_at(self) -> datetime:
        return self.inserted_at

    @property
    def job_updated_at(self) -> datetime:
        return self.updated_at


class DocTransformJobCreate(SQLModel):
    source_document_id: UUID

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportAssignmentType]


class DocTransformJobUpdate(SQLModel):
    transformed_document_id: UUID | None = None
    task_id: str | None = None
    status: TransformationStatus | None = None
    error_message: str | None = None
    trace_id: str | None = None
