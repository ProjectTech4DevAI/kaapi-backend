"""Assessment DB tables, shared enums, and legacy Assessment Run UI models.

Tables + ``AssessmentStatus``/``AssessmentMethod`` enums are shared by every method; the
rest is the legacy RUN surface. API-client models live in ``assessment_api.py``.
"""

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import Column, Index, Text, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field as SQLField
from sqlmodel import Relationship, SQLModel

from app.core.util import now

if TYPE_CHECKING:
    from app.models.batch_job import BatchJob
    from app.models.job import Job


def _pg_enum(enum_cls: type[StrEnum], name: str) -> postgresql.ENUM:
    """Postgres ENUM bound to a StrEnum's values; type created by migration."""
    return postgresql.ENUM(
        enum_cls,
        name=name,
        values_callable=lambda e: [m.value for m in e],
        create_type=False,
    )


class AssessmentMethod(StrEnum):
    RESPONSE = "RESPONSE"
    BATCH = "BATCH"
    RUN = "RUN"


class AssessmentStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
    FAILED = "FAILED"

    @classmethod
    def terminal(cls) -> frozenset["AssessmentStatus"]:
        """Finished statuses: no further transitions, so the result is final."""
        return frozenset({cls.COMPLETED, cls.COMPLETED_WITH_ERRORS, cls.FAILED})


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class Stage(StrEnum):
    """Legacy RUN pipeline stages, in order."""

    PRE_FILTER_TOPIC_RELEVANCE = "PRE_FILTER_TOPIC_RELEVANCE"
    PRE_FILTER_DUPLICATE_DETECTION = "PRE_FILTER_DUPLICATE_DETECTION"
    L2_ASSESSMENT = "L2_ASSESSMENT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class StageStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AssessmentConfigRef(BaseModel):
    """Pin to a saved config version; shared by the API create and the legacy run create."""

    id: UUID
    version: int = Field(ge=1)


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentAttachment(BaseModel):
    """External-dataset attachment column config (RUN / BATCH-by-ref)."""

    column: str = Field(..., description="Dataset column holding the attachment")
    type: Literal["image", "pdf", "mixed"] = Field(
        ...,
        description="'image'/'pdf' fix the type; 'mixed' resolves per-row via type_column",
    )
    format: Literal["url", "base64"] = Field(..., description="Data format")
    type_column: str | None = Field(
        None, description="'mixed' only: column whose value decides each row's type"
    )
    type_value_map: dict[str, Literal["image", "pdf"]] | None = Field(
        None, description="'mixed' only: maps a type_column value to 'image' or 'pdf'"
    )

    @model_validator(mode="after")
    def _validate_mixed(self) -> "AssessmentAttachment":
        if self.type == "mixed" and (
            self.type_column is None or not self.type_value_map
        ):
            raise ValueError(
                "type='mixed' requires 'type_column' and a non-empty 'type_value_map'."
            )
        return self


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class InputBinding(BaseModel):
    """External-dataset column mapping + prompt template."""

    prompt: str = Field(
        ..., min_length=1, description="User-prompt template with {column} placeholders"
    )
    text_columns: list[str] = Field(default_factory=list)
    attachments: list[AssessmentAttachment] = Field(default_factory=list)


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class RunExecution(BaseModel):
    """Legacy RUN pipeline runtime state (assessment_run.execution)."""

    stage: Stage | None = None
    stage_status: StageStatus | None = None
    pipeline: dict[str, Any] | None = None
    stage_batches: dict[str, int] | None = None
    prefilter_total_rows: int | None = None
    prefilter_total_passed: int | None = None
    prefilter_total_rejected: int | None = None
    object_store_url: str | None = None
    prefilter_object_store_url: str | None = None


class Assessment(SQLModel, table=True):
    """Parent — one submission: method + shared data source."""

    __tablename__ = "assessment"
    __table_args__ = (
        Index(
            "idx_assessment_org_project", "organization_id", "project_id", "inserted_at"
        ),
        Index("idx_assessment_status", "status"),
        Index("idx_assessment_method", "method"),
        Index(
            "idx_assessment_job", "job_id", postgresql_where=text("job_id IS NOT NULL")
        ),
    )

    id: UUID = SQLField(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the assessment"},
    )
    experiment_name: str | None = SQLField(
        default=None,
        index=True,
        sa_column_kwargs={"comment": "Experiment name; required for BATCH/RUN"},
    )
    method: AssessmentMethod = SQLField(
        sa_column=Column(
            _pg_enum(AssessmentMethod, "assessment_method"), nullable=False
        ),
    )
    status: AssessmentStatus = SQLField(
        sa_column=Column(
            _pg_enum(AssessmentStatus, "assessment_status"),
            nullable=False,
            server_default=AssessmentStatus.PENDING,
        ),
    )

    job_id: UUID | None = SQLField(
        default=None,
        foreign_key="job.id",
        nullable=True,
        ondelete="SET NULL",
        sa_column_kwargs={
            "comment": "RESPONSE execution job; resolve call/chain via job_id"
        },
    )

    input: dict[str, Any] | None = SQLField(
        default=None,
        sa_column=Column(
            JSONB,
            nullable=True,
            comment="Method-shaped: ResponseInput (RESPONSE) / BatchInput (BATCH) / InputBinding (RUN)",
        ),
    )
    # NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
    dataset_id: int | None = SQLField(
        default=None,
        foreign_key="evaluation_dataset.id",
        nullable=True,
        ondelete="SET NULL",
        sa_column_kwargs={
            "comment": "External dataset (RUN); binding lives in `input`"
        },
    )

    organization_id: int = SQLField(
        foreign_key="organization.id", nullable=False, ondelete="CASCADE"
    )
    project_id: int = SQLField(
        foreign_key="project.id", nullable=False, ondelete="CASCADE"
    )
    inserted_at: datetime = SQLField(default_factory=now, nullable=False)
    updated_at: datetime = SQLField(default_factory=now, nullable=False)

    job: Optional["Job"] = Relationship(
        sa_relationship_kwargs={"foreign_keys": "[Assessment.job_id]"}
    )


class AssessmentRun(SQLModel, table=True):
    """Child — one config execution (BATCH + RUN; RESPONSE has no run)."""

    __tablename__ = "assessment_run"
    __table_args__ = (
        Index("idx_assessment_run_assessment_id", "assessment_id"),
        Index("idx_assessment_run_config", "config_id", "config_version"),
        Index("idx_assessment_run_status", "status"),
    )

    id: int | None = SQLField(default=None, primary_key=True)
    assessment_id: UUID = SQLField(
        foreign_key="assessment.id", nullable=False, ondelete="CASCADE"
    )

    config_id: UUID = SQLField(foreign_key="config.id", nullable=False)
    config_version: int = SQLField(nullable=False)

    status: AssessmentStatus = SQLField(
        sa_column=Column(
            _pg_enum(AssessmentStatus, "assessment_status"),
            nullable=False,
            server_default=AssessmentStatus.PENDING,
        ),
    )
    total_items: int = SQLField(default=0, nullable=False)

    batch_job_id: int | None = SQLField(
        default=None,
        foreign_key="batch_job.id",
        nullable=True,
        ondelete="SET NULL",
        sa_column_kwargs={
            "comment": "BATCH job; RUN tracks its batches inside execution"
        },
    )
    # Staged-batch runtime bag (RunExecution shape). Used by the legacy RUN UI
    # pipeline AND by the new BATCH API-client path when the config carries
    # pre-filters — both drive stage/verdict/gate state through this JSONB.
    execution: dict[str, Any] | None = SQLField(
        default=None,
        sa_column=Column(
            JSONB,
            nullable=True,
            comment="Staged-batch runtime (RunExecution): RUN + BATCH-with-prefilters",
        ),
    )
    post_processing_config: dict[str, Any] | None = SQLField(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    error_message: str | None = SQLField(
        default=None, sa_column=Column(Text, nullable=True)
    )
    inserted_at: datetime = SQLField(default_factory=now, nullable=False)
    updated_at: datetime = SQLField(default_factory=now, nullable=False)

    assessment: Optional["Assessment"] = Relationship(
        sa_relationship_kwargs={"foreign_keys": "[AssessmentRun.assessment_id]"}
    )
    batch_job: Optional["BatchJob"] = Relationship(
        sa_relationship_kwargs={"foreign_keys": "[AssessmentRun.batch_job_id]"}
    )


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentExecutionPublic(BaseModel):
    config_id: UUID
    config_version: int
    status: AssessmentStatus
    total_items: int
    error_message: str | None = None
    inserted_at: datetime
    updated_at: datetime


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentResponse(BaseModel):
    assessment_id: UUID
    executions: list[AssessmentExecutionPublic] = []


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentPublic(BaseModel):
    id: UUID
    experiment_name: str | None = None
    status: AssessmentStatus
    executions: list[AssessmentExecutionPublic] = []
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentExportRow(BaseModel):
    """Flattened result row for CSV/XLSX/JSON export."""

    assessment_id: UUID
    experiment_name: str | None
    execution_id: int
    execution_status: str
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


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentRunCreate(BaseModel):
    experiment_name: str
    dataset_id: int
    input_binding: InputBinding
    configs: list[AssessmentConfigRef] = Field(min_length=1, max_length=4)
    post_processing_config: dict[str, Any] | None = None


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentRunCounts(BaseModel):
    total: int = 0
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentRunStat(BaseModel):
    run_id: int
    config_id: UUID | None
    config_version: int | None
    status: AssessmentStatus
    total_items: int
    error_message: str | None = None
    updated_at: datetime | None = None


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentRunSummary(BaseModel):
    run_id: int
    assessment_id: UUID
    config_id: UUID
    config_version: int
    status: AssessmentStatus


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentRunPublic(BaseModel):
    id: int
    assessment_id: UUID
    config_id: UUID
    config_version: int
    status: AssessmentStatus
    total_items: int
    batch_job_id: int | None = None
    execution: RunExecution | None = None
    post_processing_config: dict[str, Any] | None = None
    error_message: str | None = None
    inserted_at: datetime
    updated_at: datetime


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentRunResponse(BaseModel):
    assessment_id: UUID
    experiment_name: str | None = None
    dataset_id: int | None = None
    dataset_name: str | None = None
    num_configs: int
    runs: list[AssessmentRunSummary] = []


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentRunOverview(BaseModel):
    id: UUID
    experiment_name: str | None = None
    status: AssessmentStatus
    dataset_id: int | None = None
    dataset_name: str | None = None
    input_binding: InputBinding | None = None
    counts: AssessmentRunCounts = AssessmentRunCounts()
    run_stats: list[AssessmentRunStat] = []
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentDatasetPreview(BaseModel):
    headers: list[str]
    rows: list[list[str]]
    returned_rows: int = 0
    truncated: bool = False


# NOTE: Legacy, this is for Assessment Run UI only. The new Assessment pipeline does not use this.
class AssessmentDatasetResponse(BaseModel):
    dataset_id: int
    dataset_name: str
    description: str | None = None
    total_items: int = 0
    file_extension: str | None = None
    object_store_url: str | None = None
    signed_url: str | None = None
    preview: AssessmentDatasetPreview | None = None
