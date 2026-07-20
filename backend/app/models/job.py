from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from app.core.util import now


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class JobType(str, Enum):
    RESPONSE = "RESPONSE"
    LLM_API = "LLM_API"
    LLM_CHAIN = "LLM_CHAIN"
    LLM_GUARDRAILS = "LLM_GUARDRAILS"
    PROMPT_IMPROVEMENT = "PROMPT_IMPROVEMENT"


class Job(SQLModel, table=True):
    """Database model for tracking async jobs."""

    __tablename__ = "job"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the job"},
    )
    task_id: str | None = Field(
        nullable=True,
        description="Celery task ID returned when job is queued.",
        sa_column_kwargs={"comment": "Celery task ID returned when job is queued"},
    )
    trace_id: str | None = Field(
        default=None,
        description="Tracing ID for correlating logs and traces.",
        sa_column_kwargs={"comment": "Tracing ID for correlating logs and traces"},
    )
    project_id: int | None = Field(
        default=None,
        foreign_key="project.id",
        ondelete="CASCADE",
        index=True,
        description="Project ID of the project the job belongs to.",
        sa_column_kwargs={"comment": "Project ID of the job's project"},
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if the job fails.",
        sa_column_kwargs={"comment": "Error details if the job fails"},
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING,
        description="Current state of the job.",
        sa_column_kwargs={
            "comment": "Current state of the job (PENDING, PROCESSING, SUCCESS, FAILED)"
        },
    )
    job_type: JobType = Field(
        description="Type of job being executed (e.g., response, ingestion).",
        sa_column_kwargs={
            "comment": "Type of job being executed (e.g., RESPONSE, LLM_API, LLM_CHAIN, LLM_GUARDRAILS, PROMPT_IMPROVEMENT)"
        },
    )
    meta: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(
            JSONB,
            nullable=True,
            comment=(
                "Per-job-type tracking payload. For LLM_GUARDRAILS this stores "
                "{'request': {...}, 'response': {...}} capturing the inbound "
                "guardrails request and the upstream guardrails service response."
            ),
        ),
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


class JobUpdate(SQLModel):
    status: JobStatus | None = None
    error_message: str | None = None
    task_id: str | None = None
    meta: dict[str, Any] | None = None
