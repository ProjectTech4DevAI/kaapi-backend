from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from sqlmodel import Column, Field, SQLModel, Text

from app.core.util import now
from app.models.collection import CollectionIDPublic, CollectionPublic


class CollectionJobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESSFUL = "SUCCESSFUL"
    FAILED = "FAILED"


class CollectionActionType(str, Enum):
    CREATE = "CREATE"
    DELETE = "DELETE"


class CollectionJob(SQLModel, table=True):
    """Database model for CollectionJob operations."""

    __tablename__ = "collection_jobs"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the collection job"},
    )
    status: CollectionJobStatus = Field(
        default=CollectionJobStatus.PENDING,
        nullable=False,
        description="Current job status",
        sa_column_kwargs={
            "comment": "Current job status (PENDING, PROCESSING, SUCCESSFUL, FAILED)"
        },
    )
    action_type: CollectionActionType = Field(
        nullable=False,
        description="Type of operation",
        sa_column_kwargs={"comment": "Type of operation (CREATE, DELETE)"},
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
    docs_num: int | None = Field(
        description="Total number of documents to be processed in this job",
        sa_column_kwargs={
            "comment": "Total number of documents to be processed in this job"
        },
    )
    total_size: int | None = Field(
        description="Total size of documents being uploaded to collection",
        sa_column_kwargs={
            "comment": "Total size of documents being uploaded to collection"
        },
    )
    error_message: str | None = Field(
        default=None,
        sa_column=Column(
            Text, nullable=True, comment="Error message if the job failed"
        ),
    )

    # Foreign keys
    collection_id: UUID | None = Field(
        foreign_key="collection.id",
        nullable=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the collection"},
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
        nullable=False,
        description="When the job record was created",
        sa_column_kwargs={"comment": "Timestamp when the job was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        description="Last time the job record was updated",
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


# Request models
class CollectionJobCreate(SQLModel):
    collection_id: UUID | None = None
    status: CollectionJobStatus
    action_type: CollectionActionType
    batch_size: int | None = None
    num_docs: int | None = None
    total_size: int | None = None
    project_id: int


class CollectionJobUpdate(SQLModel):
    task_id: str | None = None
    status: CollectionJobStatus | None = None
    error_message: str | None = None
    collection_id: UUID | None = None
    trace_id: str | None = None


##Response models
class CollectionJobBasePublic(SQLModel):
    job_id: UUID
    status: CollectionJobStatus


class CollectionJobImmediatePublic(CollectionJobBasePublic):
    job_inserted_at: datetime
    job_updated_at: datetime


class CollectionJobPublic(CollectionJobBasePublic):
    action_type: CollectionActionType
    collection: CollectionPublic | CollectionIDPublic | None = None
    error_message: str | None = None
