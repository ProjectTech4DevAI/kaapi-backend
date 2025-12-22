from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import Column, Index, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field as SQLField
from sqlmodel import Relationship, SQLModel

from app.core.util import now

if TYPE_CHECKING:
    from .batch_job import BatchJob
    from .organization import Organization
    from .project import Project


class DatasetItem(BaseModel):
    """Model for a single dataset item (Q&A pair)."""

    question: str = Field(..., description="The question/input")
    answer: str = Field(..., description="The expected answer/output")


class DatasetUploadResponse(BaseModel):
    """Response model for dataset upload."""

    dataset_id: int = Field(..., description="Database ID of the created dataset")
    dataset_name: str = Field(..., description="Name of the created dataset")
    total_items: int = Field(
        ..., description="Total number of items uploaded (after duplication)"
    )
    original_items: int = Field(
        ..., description="Number of original items before duplication"
    )
    duplication_factor: int = Field(
        default=5, description="Number of times each item was duplicated"
    )
    langfuse_dataset_id: str | None = Field(
        None, description="Langfuse dataset ID if available"
    )
    object_store_url: str | None = Field(
        None, description="Object store URL if uploaded"
    )


class EvaluationResult(BaseModel):
    """Model for a single evaluation result."""

    input: str = Field(..., description="The input question/prompt used for evaluation")
    output: str = Field(..., description="The actual output from the assistant")
    expected: str = Field(..., description="The expected output from the dataset")
    response_id: str | None = Field(None, description="ID from the batch response body")


class Experiment(BaseModel):
    """Model for the complete experiment evaluation response."""

    experiment_name: str = Field(..., description="Name of the experiment")
    dataset_name: str = Field(
        ..., description="Name of the dataset used for evaluation"
    )
    results: list[EvaluationResult] = Field(
        ..., description="List of evaluation results"
    )
    total_items: int = Field(..., description="Total number of items evaluated")
    note: str = Field(..., description="Additional notes about the evaluation process")


# Database Models


class EvaluationDataset(SQLModel, table=True):
    """Database table for evaluation datasets."""

    __tablename__ = "evaluation_dataset"
    __table_args__ = (
        UniqueConstraint(
            "name",
            "organization_id",
            "project_id",
            name="uq_evaluation_dataset_name_org_project",
        ),
    )

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the dataset"},
    )

    # Dataset information
    name: str = SQLField(
        index=True,
        description="Name of the dataset",
        sa_column_kwargs={"comment": "Name of the evaluation dataset"},
    )
    description: str | None = SQLField(
        default=None,
        description="Optional description of the dataset",
        sa_column_kwargs={"comment": "Description of the dataset"},
    )

    # Dataset metadata stored as JSONB
    dataset_metadata: dict[str, Any] = SQLField(
        default_factory=dict,
        sa_column=Column(
            JSONB,
            nullable=False,
            comment="Dataset metadata (item counts, duplication factor, etc.)",
        ),
        description=(
            "Dataset metadata (original_items_count, total_items_count, "
            "duplication_factor)"
        ),
    )

    # Storage references
    object_store_url: str | None = SQLField(
        default=None,
        description="Object store URL where CSV is stored",
        sa_column_kwargs={"comment": "S3 URL where the dataset CSV is stored"},
    )
    langfuse_dataset_id: str | None = SQLField(
        default=None,
        description="Langfuse dataset ID for reference",
        sa_column_kwargs={
            "comment": "Langfuse dataset ID for observability integration"
        },
    )

    # Foreign keys
    organization_id: int = SQLField(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = SQLField(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )

    # Timestamps
    inserted_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the evaluation dataset was created"
        },
    )
    updated_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the evaluation dataset was last updated"
        },
    )

    # Relationships
    project: "Project" = Relationship()
    organization: "Organization" = Relationship()
    evaluation_runs: list["EvaluationRun"] = Relationship(
        back_populates="evaluation_dataset"
    )


class EvaluationRun(SQLModel, table=True):
    """Database table for evaluation runs."""

    __tablename__ = "evaluation_run"
    __table_args__ = (
        Index("idx_eval_run_status_org", "status", "organization_id"),
        Index("idx_eval_run_status_project", "status", "project_id"),
    )

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the evaluation run"},
    )

    # Input fields (provided by user)
    run_name: str = SQLField(
        index=True,
        description="Name of the evaluation run",
        sa_column_kwargs={"comment": "Name of the evaluation run"},
    )
    dataset_name: str = SQLField(
        description="Name of the Langfuse dataset",
        sa_column_kwargs={"comment": "Name of the Langfuse dataset used"},
    )

    config_id: UUID = SQLField(
        foreign_key="config.id",
        nullable=True,
        description="Reference to the stored config used for this evaluation",
        sa_column_kwargs={"comment": "Reference to the stored config used"},
    )
    config_version: int = SQLField(
        nullable=True,
        ge=1,
        description="Version of the config used for this evaluation",
        sa_column_kwargs={"comment": "Version of the config used"},
    )

    # Dataset reference
    dataset_id: int = SQLField(
        foreign_key="evaluation_dataset.id",
        nullable=False,
        ondelete="CASCADE",
        description="Reference to the evaluation_dataset used for this run",
        sa_column_kwargs={"comment": "Reference to the evaluation dataset"},
    )

    # Batch job references
    batch_job_id: int | None = SQLField(
        default=None,
        foreign_key="batch_job.id",
        ondelete="SET NULL",
        description=(
            "Reference to the batch_job that processes this evaluation (responses)"
        ),
        sa_column_kwargs={"comment": "Reference to the batch job for responses"},
    )
    embedding_batch_job_id: int | None = SQLField(
        default=None,
        foreign_key="batch_job.id",
        nullable=True,
        ondelete="SET NULL",
        description="Reference to the batch_job for embedding-based similarity scoring",
        sa_column_kwargs={
            "comment": "Reference to the batch job for embedding similarity scoring"
        },
    )

    # Output/Status fields (updated by system during processing)
    status: str = SQLField(
        default="pending",
        description="Overall evaluation status: pending, processing, completed, failed",
        sa_column_kwargs={
            "comment": "Evaluation status (pending, processing, completed, failed)"
        },
    )
    object_store_url: str | None = SQLField(
        default=None,
        description="Object store URL of processed evaluation results for future reference",
        sa_column_kwargs={"comment": "S3 URL of processed evaluation results"},
    )
    total_items: int = SQLField(
        default=0,
        description="Total number of items evaluated (set during processing)",
        sa_column_kwargs={"comment": "Total number of items evaluated"},
    )

    # Score field - dict requires sa_column
    score: dict[str, Any] | None = SQLField(
        default=None,
        sa_column=Column(
            JSONB,
            nullable=True,
            comment="Evaluation scores (correctness, cosine_similarity, etc.)",
        ),
        description="Evaluation scores (e.g., correctness, cosine_similarity, etc.)",
    )

    # Error message field
    error_message: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text, nullable=True, comment="Error message if evaluation failed"
        ),
        description="Error message if failed",
    )

    # Foreign keys
    organization_id: int = SQLField(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = SQLField(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        description="The timestamp when the evaluation run was started",
        sa_column_kwargs={"comment": "Timestamp when the evaluation run was started"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        description="The timestamp when the evaluation run was last updated",
        sa_column_kwargs={
            "comment": "Timestamp when the evaluation run was last updated"
        },
    )

    # Relationships
    project: "Project" = Relationship()
    organization: "Organization" = Relationship()
    evaluation_dataset: "EvaluationDataset" = Relationship(
        back_populates="evaluation_runs"
    )
    batch_job: Optional["BatchJob"] = Relationship(
        sa_relationship_kwargs={"foreign_keys": "[EvaluationRun.batch_job_id]"}
    )
    embedding_batch_job: Optional["BatchJob"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[EvaluationRun.embedding_batch_job_id]"
        }
    )


class EvaluationRunCreate(SQLModel):
    """Model for creating an evaluation run."""

    run_name: str = Field(description="Name of the evaluation run", min_length=3)
    dataset_id: int = Field(description="ID of the evaluation dataset")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Evaluation configuration (flexible dict with llm, instructions, "
            "vector_store_ids, etc.)"
        ),
    )


class EvaluationRunPublic(SQLModel):
    """Public model for evaluation runs."""

    id: int
    run_name: str
    dataset_name: str
    config_id: UUID | None
    config_version: int | None
    dataset_id: int
    batch_job_id: int | None
    embedding_batch_job_id: int | None
    status: str
    object_store_url: str | None
    total_items: int
    score: dict[str, Any] | None
    error_message: str | None
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class EvaluationDatasetCreate(SQLModel):
    """Model for creating an evaluation dataset."""

    name: str = Field(description="Name of the dataset", min_length=1)
    description: str | None = Field(None, description="Optional dataset description")
    dataset_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Dataset metadata (original_items_count, total_items_count, "
            "duplication_factor)"
        ),
    )
    object_store_url: str | None = Field(
        None, description="Object store URL where CSV is stored"
    )
    langfuse_dataset_id: str | None = Field(
        None, description="Langfuse dataset ID for reference"
    )


class EvaluationDatasetPublic(SQLModel):
    """Public model for evaluation datasets."""

    id: int
    name: str
    description: str | None
    dataset_metadata: dict[str, Any]
    object_store_url: str | None
    langfuse_dataset_id: str | None
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime
