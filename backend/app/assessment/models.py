"""Assessment models — DB table, Pydantic schemas, and LLM param wrappers."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import Column, Index, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field as SQLField
from sqlmodel import Relationship, SQLModel

from app.core.util import now
from app.models.llm.request import TextLLMParams

if TYPE_CHECKING:
    from app.models.batch_job import BatchJob


# ── Database models ─────────────────────────────────────────────


class Assessment(SQLModel, table=True):
    """Manager table for multi-config assessment evaluations."""

    __tablename__ = "assessment"
    __table_args__ = (
        Index("idx_assessment_status_org", "status", "organization_id"),
        Index("idx_assessment_status_project", "status", "project_id"),
    )

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the assessment"},
    )
    experiment_name: str = SQLField(
        index=True,
        description="Experiment name shared by child config runs",
        sa_column_kwargs={"comment": "Experiment name shared by child config runs"},
    )
    dataset_id: int = SQLField(
        foreign_key="evaluation_dataset.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the evaluation dataset"},
    )
    dataset_name: str = SQLField(
        nullable=False,
        description="Name of the dataset used by this assessment",
        sa_column_kwargs={"comment": "Name of the dataset used by this assessment"},
    )
    status: str = SQLField(
        default="pending",
        description="Overall assessment status across all child evaluation runs",
        sa_column_kwargs={
            "comment": "Overall assessment status across all child evaluation runs"
        },
    )
    total_runs: int = SQLField(
        default=0,
        nullable=False,
        sa_column_kwargs={"comment": "Total number of child evaluation runs"},
    )
    pending_runs: int = SQLField(
        default=0,
        nullable=False,
        sa_column_kwargs={"comment": "Number of child runs in pending state"},
    )
    processing_runs: int = SQLField(
        default=0,
        nullable=False,
        sa_column_kwargs={"comment": "Number of child runs in processing state"},
    )
    completed_runs: int = SQLField(
        default=0,
        nullable=False,
        sa_column_kwargs={"comment": "Number of child runs in completed state"},
    )
    failed_runs: int = SQLField(
        default=0,
        nullable=False,
        sa_column_kwargs={"comment": "Number of child runs in failed state"},
    )
    run_stats: list[dict[str, Any]] = SQLField(
        default_factory=list,
        sa_column=Column(
            JSONB,
            nullable=False,
            comment="Cached status snapshot for child evaluation runs",
        ),
        description="Cached status snapshot for child evaluation runs",
    )
    error_message: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Aggregated error message for child run failures",
        ),
        description="Aggregated error message for child run failures",
    )
    callback_url: str | None = SQLField(
        default=None,
        nullable=True,
        sa_column_kwargs={
            "comment": "Optional frontend callback URL for status updates"
        },
    )
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
    inserted_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the assessment was created"},
    )
    updated_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the assessment was last updated"},
    )


class AssessmentRun(SQLModel, table=True):
    """Dedicated table for assessment evaluation runs."""

    __tablename__ = "assessment_run"
    __table_args__ = (
        Index("idx_assessment_run_status_org", "status", "organization_id"),
        Index("idx_assessment_run_status_project", "status", "project_id"),
        Index("idx_assessment_run_assessment_id", "assessment_id"),
    )

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the assessment run"},
    )
    run_name: str = SQLField(
        index=True,
        description="Name of the assessment run (matches experiment name)",
        sa_column_kwargs={"comment": "Name of the assessment run"},
    )
    assessment_id: int | None = SQLField(
        default=None,
        foreign_key="assessment.id",
        nullable=True,
        ondelete="SET NULL",
        sa_column_kwargs={"comment": "Reference to parent assessment manager row"},
    )
    dataset_id: int = SQLField(
        foreign_key="evaluation_dataset.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the evaluation dataset"},
    )
    dataset_name: str = SQLField(
        nullable=False,
        sa_column_kwargs={"comment": "Name of the dataset used"},
    )
    config_id: UUID | None = SQLField(
        default=None,
        foreign_key="config.id",
        nullable=True,
        sa_column_kwargs={"comment": "Reference to the stored config used"},
    )
    config_version: int | None = SQLField(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Version of the config used"},
    )
    status: str = SQLField(
        default="pending",
        sa_column_kwargs={
            "comment": "Run status: pending, processing, completed, failed"
        },
    )
    batch_job_id: int | None = SQLField(
        default=None,
        foreign_key="batch_job.id",
        nullable=True,
        ondelete="SET NULL",
        sa_column_kwargs={"comment": "Reference to the batch job processing this run"},
    )
    total_items: int = SQLField(
        default=0,
        nullable=False,
        sa_column_kwargs={"comment": "Total number of dataset items in this run"},
    )
    input: dict[str, Any] | None = SQLField(
        default=None,
        sa_column=Column(
            JSONB,
            nullable=True,
            comment="Assessment input config: prompt_template, text_columns, attachments, output_schema",
        ),
    )
    object_store_url: str | None = SQLField(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "S3 URL of processed batch results"},
    )
    error_message: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Error message if the run failed",
        ),
    )
    eval_score: dict[str, Any] | None = SQLField(
        default=None,
        sa_column=Column(
            JSONB,
            nullable=True,
            comment="Evaluation scores (reserved for future use)",
        ),
    )
    eval_score_trace_url: str | None = SQLField(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "S3 URL for evaluation score traces (reserved)"},
    )
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
    inserted_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the run was created"},
    )
    updated_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the run was last updated"},
    )

    # Relationships
    batch_job: Optional["BatchJob"] = Relationship(
        sa_relationship_kwargs={"foreign_keys": "[AssessmentRun.batch_job_id]"}
    )


class AssessmentPublic(BaseModel):
    """Public model for assessment manager rows."""

    id: int
    experiment_name: str
    dataset_id: int
    dataset_name: str
    status: str
    total_runs: int
    pending_runs: int
    processing_runs: int
    completed_runs: int
    failed_runs: int
    run_stats: list[dict[str, Any]]
    error_message: str | None
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


# ── Extended LLM params ──────────────────────────────────────────


class AssessmentTextLLMParams(TextLLMParams):
    """TextLLMParams extended with response_format and output_schema for assessments."""

    response_format: Literal["text", "json_object"] = Field(
        default="text",
        description="Response format: 'text' or 'json_object'",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema for structured output",
    )


# ── Attachment / config references ───────────────────────────────


class AssessmentAttachment(BaseModel):
    """Attachment column configuration."""

    column: str = Field(..., description="Column name containing the attachment data")
    type: str = Field(..., description="Attachment type: 'image' or 'pdf'")
    format: str = Field(..., description="Data format: 'url' or 'base64'")


class AssessmentConfigRef(BaseModel):
    """Reference to a stored config version."""

    config_id: UUID = Field(..., description="Stored config UUID")
    config_version: int = Field(..., ge=1, description="Config version number")


# ── Request model ────────────────────────────────────────────────


class AssessmentCreate(BaseModel):
    """Request body for creating an assessment evaluation."""

    experiment_name: str = Field(
        ..., min_length=1, description="Name for this evaluation experiment"
    )
    dataset_id: int = Field(..., description="ID of the uploaded dataset")
    prompt_template: str | None = Field(
        None,
        description=(
            "Prompt template with {column} placeholders. "
            "If null, all text columns are concatenated."
        ),
    )
    text_columns: list[str] = Field(
        default_factory=list, description="Column names mapped as text input"
    )
    attachments: list[AssessmentAttachment] = Field(
        default_factory=list, description="Attachment column configurations"
    )
    output_schema: dict[str, Any] | None = Field(
        None, description="JSON Schema for structured output"
    )
    configs: list[AssessmentConfigRef] = Field(
        ..., min_length=1, max_length=4, description="Config versions to evaluate"
    )


# ── Response models ──────────────────────────────────────────────


class AssessmentRunSummary(BaseModel):
    """Summary of a single evaluation run created for one config."""

    run_id: int
    assessment_id: int | None = None
    config_id: str
    config_version: int
    status: str


class AssessmentResponse(BaseModel):
    """Response after submitting an assessment evaluation."""

    assessment_id: int
    experiment_name: str
    dataset_id: int
    dataset_name: str
    num_configs: int
    runs: list[AssessmentRunSummary]


class AssessmentRunPublic(BaseModel):
    """Public view of an assessment evaluation run."""

    id: int
    assessment_id: int | None
    run_name: str
    dataset_name: str
    dataset_id: int
    config_id: UUID | None
    config_version: int | None
    status: str
    total_items: int
    error_message: str | None
    organization_id: int
    project_id: int
    input: dict[str, Any] | None = Field(
        None,
        description="Assessment input config (prompt_template, text_columns, attachments, output_schema)",
    )
    inserted_at: datetime
    updated_at: datetime


class AssessmentExportRow(BaseModel):
    """Flattened assessment result row for CSV/XLSX export."""

    assessment_id: int
    experiment_name: str
    dataset_id: int | None
    dataset_name: str | None
    run_id: int
    run_name: str
    run_status: str
    config_id: UUID | None
    config_version: int | None
    row_id: str
    result_status: str
    input_data: dict[str, str] | None = None
    output: str | None = None
    error: str | None = None
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    updated_at: datetime


class AssessmentDatasetResponse(BaseModel):
    """Response model for assessment dataset."""

    dataset_id: int
    dataset_name: str
    description: str | None = None
    total_items: int = 0
    file_extension: str | None = None
    object_store_url: str | None = None
    signed_url: str | None = None
