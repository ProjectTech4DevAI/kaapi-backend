from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Column, Index, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel

from app.core.util import now


class BatchJobType(str, Enum):
    """Type of batch job being executed."""

    EVALUATION = "evaluation"
    STT_EVALUATION = "stt_evaluation"
    TTS_EVALUATION = "tts_evaluation"
    EMBEDDING = "embedding"


if TYPE_CHECKING:
    from .organization import Organization
    from .project import Project


class BatchJob(SQLModel, table=True):
    """Database model for BatchJob operations."""

    __tablename__ = "batch_job"
    __table_args__ = (
        Index("idx_batch_job_status_org", "provider_status", "organization_id"),
        Index("idx_batch_job_status_project", "provider_status", "project_id"),
    )

    id: int | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the batch job"},
    )

    # Provider and job type
    provider: str = Field(
        description="LLM provider name (e.g., 'openai', 'anthropic')",
        sa_column_kwargs={"comment": "LLM provider name (e.g., openai, anthropic)"},
    )
    job_type: str = Field(
        index=True,
        description=(
            "Type of batch job (e.g., 'evaluation', 'classification', 'embedding')"
        ),
        sa_column_kwargs={
            "comment": "Type of batch job (e.g., evaluation, classification, embedding)"
        },
    )

    # Batch configuration - stores all provider-specific config
    config: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(
            JSONB,
            nullable=False,
            comment="Complete batch configuration including model, temperature, instructions, tools, etc.",
        ),
        description=(
            "Complete batch configuration including model, temperature, instructions, tools, etc."
        ),
    )

    # Provider-specific batch tracking
    provider_batch_id: str | None = Field(
        default=None,
        description="Provider's batch job ID (e.g., OpenAI batch_id)",
        sa_column_kwargs={"comment": "Provider's batch job ID (e.g., OpenAI batch_id)"},
    )
    provider_file_id: str | None = Field(
        default=None,
        description="Provider's input file ID",
        sa_column_kwargs={"comment": "Provider's input file ID"},
    )
    provider_output_file_id: str | None = Field(
        default=None,
        description="Provider's output file ID",
        sa_column_kwargs={"comment": "Provider's output file ID"},
    )

    # Provider status tracking
    provider_status: str | None = Field(
        default=None,
        description=(
            "Provider-specific status (e.g., OpenAI: validating, in_progress, "
            "finalizing, completed, failed, expired, cancelling, cancelled)"
        ),
        sa_column_kwargs={
            "comment": "Provider-specific status (e.g., validating, in_progress, completed, failed)"
        },
    )

    # Raw results (before parent-specific processing)
    raw_output_url: str | None = Field(
        default=None,
        description="S3 URL of raw batch output file",
        sa_column_kwargs={"comment": "S3 URL of raw batch output file"},
    )
    total_items: int = Field(
        default=0,
        description="Total number of items in the batch",
        sa_column_kwargs={"comment": "Total number of items in the batch"},
    )

    # Error handling
    error_message: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True, comment="Error message if batch failed"),
        description="Error message if batch failed",
    )

    # Foreign keys
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        index=True,
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        index=True,
        sa_column_kwargs={"comment": "Reference to the project"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        description="The timestamp when the batch job was started",
        sa_column_kwargs={"comment": "Timestamp when the batch job was started"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        description="The timestamp when the batch job was last updated",
        sa_column_kwargs={"comment": "Timestamp when the batch job was last updated"},
    )

    # Relationships
    organization: Optional["Organization"] = Relationship()
    project: Optional["Project"] = Relationship()


class BatchJobCreate(SQLModel):
    """Schema for creating a new batch job."""

    provider: str
    job_type: str
    config: dict[str, Any] = Field(default_factory=dict)
    provider_batch_id: str | None = None
    provider_file_id: str | None = None
    provider_output_file_id: str | None = None
    provider_status: str | None = None
    raw_output_url: str | None = None
    total_items: int = 0
    error_message: str | None = None
    organization_id: int
    project_id: int


class BatchJobUpdate(SQLModel):
    """Schema for updating a batch job."""

    provider_batch_id: str | None = None
    provider_file_id: str | None = None
    provider_output_file_id: str | None = None
    provider_status: str | None = None
    raw_output_url: str | None = None
    total_items: int | None = None
    error_message: str | None = None


class BatchJobPublic(SQLModel):
    """Public schema for batch job responses."""

    id: int
    provider: str
    job_type: str
    config: dict[str, Any]
    provider_batch_id: str | None
    provider_file_id: str | None
    provider_output_file_id: str | None
    provider_status: str | None
    raw_output_url: str | None
    total_items: int
    error_message: str | None
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime
