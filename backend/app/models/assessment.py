"""Assessment models — DB tables, Pydantic schemas, and LLM param wrappers."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column, Index, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field as SQLField
from sqlmodel import Relationship, SQLModel

from app.core.util import now
from app.models.llm.request import TextLLMParams

if TYPE_CHECKING:
    from app.models.batch_job import BatchJob


class Assessment(SQLModel, table=True):
    """Parent assessment — one experiment over a dataset, grouping N config runs."""

    __tablename__ = "assessment"
    __table_args__ = (
        Index(
            "idx_assessment_org_project",
            "organization_id",
            "project_id",
            "inserted_at",
        ),
        Index("idx_assessment_status", "status"),
    )

    id: int | None = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the assessment"},
    )
    experiment_name: str = SQLField(
        index=True,
        sa_column_kwargs={"comment": "Name of the experiment grouping its config runs"},
    )
    dataset_id: int = SQLField(
        foreign_key="evaluation_dataset.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the evaluation dataset"},
    )
    status: str = SQLField(
        default="pending",
        sa_column_kwargs={
            "comment": (
                "Aggregate status: pending, processing, completed, "
                "completed_with_errors, failed"
            )
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
    """Child run — a single config evaluation against the parent's dataset."""

    __tablename__ = "assessment_run"
    __table_args__ = (Index("idx_assessment_run_assessment_id", "assessment_id"),)

    id: int | None = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the assessment run"},
    )
    assessment_id: int = SQLField(
        foreign_key="assessment.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the parent assessment"},
    )
    config_id: UUID = SQLField(
        foreign_key="config.id",
        nullable=False,
        sa_column_kwargs={"comment": "Reference to the stored config used"},
    )
    config_version: int = SQLField(
        nullable=False,
        sa_column_kwargs={"comment": "Version of the config used"},
    )
    status: str = SQLField(
        default="pending",
        sa_column_kwargs={
            "comment": (
                "Unified pipeline status: pending, l1_processing, l1_failed, "
                "l2_processing, completed, completed_with_errors, failed"
            )
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
    input: dict[str, Any] = SQLField(
        sa_column=Column(
            JSONB,
            nullable=False,
            comment=(
                "Assessment input: prompt_template, system_instruction, "
                "text_columns, attachments, output_schema"
            ),
        ),
    )
    object_store_url: str | None = SQLField(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "S3 URL of processed L2 batch results"},
    )
    l1_object_store_url: str | None = SQLField(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "S3 URL of stored L1 filter results JSON"},
    )
    l1_total_rows: int | None = SQLField(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Total rows fed into L1 pipeline"},
    )
    l1_total_passed: int | None = SQLField(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Rows that passed topic relevance and went to L2"},
    )
    l1_total_rejected: int | None = SQLField(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Rows rejected by topic relevance, stopped at L1"},
    )
    error_message: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Error message if the run failed",
        ),
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

    batch_job: Optional["BatchJob"] = Relationship(
        sa_relationship_kwargs={"foreign_keys": "[AssessmentRun.batch_job_id]"}
    )
    assessment: Optional["Assessment"] = Relationship(
        sa_relationship_kwargs={"foreign_keys": "[AssessmentRun.assessment_id]"}
    )


class AssessmentRunCounts(BaseModel):
    """Derived counters for a parent assessment, computed from its child runs."""

    total: int = 0
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0


class AssessmentRunStat(BaseModel):
    """Summary entry for one child run, embedded in parent responses."""

    run_id: int
    config_id: str | None
    config_version: int | None
    status: str
    total_items: int
    error_message: str | None = None
    updated_at: datetime | None = None
    l1_total_rows: int | None = None
    l1_total_passed: int | None = None
    l1_total_rejected: int | None = None


class AssessmentPublic(BaseModel):
    """Public model for a parent assessment row, with derived run aggregates."""

    id: int
    experiment_name: str
    dataset_id: int
    dataset_name: str | None = None
    status: str
    counts: AssessmentRunCounts = AssessmentRunCounts()
    run_stats: list[AssessmentRunStat] = []
    error_message: str | None = None
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class AssessmentRunPublic(BaseModel):
    """Public view of an assessment run."""

    id: int
    assessment_id: int
    experiment_name: str | None = None
    dataset_id: int | None = None
    dataset_name: str | None = None
    config_id: UUID
    config_version: int
    status: str
    total_items: int
    error_message: str | None = None
    input: dict[str, Any] | None = Field(
        None,
        description=(
            "Assessment input config: prompt_template, system_instruction, "
            "text_columns, attachments, output_schema"
        ),
    )
    l1_total_rows: int | None = None
    l1_total_passed: int | None = None
    l1_total_rejected: int | None = None
    post_processing_config: dict[str, Any] | None = None
    inserted_at: datetime
    updated_at: datetime


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


class AssessmentAttachment(BaseModel):
    """Attachment column configuration."""

    column: str = Field(..., description="Column name containing the attachment data")
    type: Literal["image", "pdf", "mixed"] = Field(
        ...,
        description=(
            "Attachment type. 'mixed' detects image vs pdf per item (for columns "
            "that contain both); 'image'/'pdf' force a type and act as fallback "
            "when per-item detection is inconclusive."
        ),
    )
    format: Literal["url", "base64"] = Field(..., description="Data format")


class AssessmentConfigRef(BaseModel):
    """Reference to a stored config version."""

    config_id: UUID = Field(..., description="Stored config UUID")
    config_version: int = Field(..., ge=1, description="Config version number")


class AssessmentCreate(BaseModel):
    """Request body for creating an assessment and child runs."""

    experiment_name: str = Field(
        ..., min_length=1, description="Name for this assessment experiment"
    )
    dataset_id: int = Field(..., description="ID of the uploaded dataset")
    prompt_template: str | None = Field(
        None,
        description=(
            "Prompt template with {column} placeholders. "
            "If null, all text columns are concatenated."
        ),
    )
    system_instruction: str | None = Field(
        None,
        description="System instruction used when generating assessment outputs",
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
        ..., min_length=1, max_length=4, description="Config versions to run"
    )
    l1_config: dict[str, Any] | None = Field(
        None,
        description=(
            "L1 pipeline config. Keys: topic_relevance (columns, prompt), "
            "duplicate_detection (columns). Omit to skip L1."
        ),
    )
    post_processing_config: dict[str, Any] | None = Field(
        None,
        description=(
            "Post-processing config applied at export. "
            "Keys: computed_columns, sort, filter."
        ),
    )


class AssessmentRunSummary(BaseModel):
    """Summary of a single assessment run created for one config."""

    run_id: int
    assessment_id: int
    config_id: str
    config_version: int
    status: str


class AssessmentResponse(BaseModel):
    """Response after submitting an assessment run request."""

    assessment_id: int
    experiment_name: str
    dataset_id: int
    dataset_name: str | None
    num_configs: int
    runs: list[AssessmentRunSummary]


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
    topic_relevance: str | None = None
    duplicate_detection: str | None = None
    output: str | None = None
    error: str | None = None
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    updated_at: datetime


class AssessmentDatasetPreview(BaseModel):
    """Lightweight preview of a dataset's columns and first N rows."""

    headers: list[str]
    rows: list[list[str]]
    returned_rows: int = 0
    truncated: bool = False


class AssessmentDatasetResponse(BaseModel):
    """Response model for assessment dataset."""

    dataset_id: int
    dataset_name: str
    description: str | None = None
    total_items: int = 0
    file_extension: str | None = None
    object_store_url: str | None = None
    signed_url: str | None = None
    preview: AssessmentDatasetPreview | None = None
