from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import field_validator
from sqlalchemy import Column, Text
from sqlmodel import Field, Relationship, SQLModel

from app.core.util import now
from app.models.project import Project


class FineTuningStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class FineTuningJobBase(SQLModel):
    base_model: str = Field(nullable=False, description="Base model for fine-tuning")
    split_ratio: float = Field(nullable=False)
    document_id: UUID = Field(foreign_key="document.id", nullable=False)
    training_file_id: str | None = Field(default=None)
    system_prompt: str = Field(sa_column=Column(Text, nullable=False))


class FineTuningJobCreate(SQLModel):
    document_id: UUID
    base_model: str
    split_ratio: list[float]
    system_prompt: str

    @field_validator("split_ratio")
    @classmethod
    def check_ratios(cls, v):
        if not v:
            raise ValueError("split_ratio cannot be empty")
        for ratio in v:
            if not (0 < ratio < 1):
                raise ValueError(
                    f"Invalid split_ratio: {ratio}. Must be between 0 and 1."
                )
        return v

    @field_validator("system_prompt")
    @classmethod
    def check_prompt(cls, v):
        if not v.strip():
            raise ValueError("system_prompt must be a non-empty string")
        return v.strip()


class FineTuning(FineTuningJobBase, table=True):
    """Database model for tracking fine-tuning jobs."""

    __tablename__ = "fine_tuning"

    id: int = Field(
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the fine-tuning job"},
    )
    base_model: str = Field(
        nullable=False,
        description="Base model for fine-tuning",
        sa_column_kwargs={"comment": "Base model used for fine-tuning"},
    )
    split_ratio: float = Field(
        nullable=False,
        sa_column_kwargs={"comment": "Train/test split ratio for the dataset"},
    )
    training_file_id: str | None = Field(
        default=None,
        sa_column_kwargs={"comment": "OpenAI training file identifier"},
    )
    system_prompt: str = Field(
        sa_column=Column(
            Text, nullable=False, comment="System prompt used during fine-tuning"
        )
    )
    provider_job_id: str | None = Field(
        default=None,
        description="Fine tuning Job ID returned by OpenAI",
        sa_column_kwargs={"comment": "Fine-tuning job ID returned by the provider"},
    )
    status: FineTuningStatus = Field(
        default=FineTuningStatus.pending,
        description="Fine tuning status",
        sa_column_kwargs={"comment": "Current status of the fine-tuning job"},
    )
    fine_tuned_model: str | None = Field(
        default=None,
        description="Final fine tuned model name from OpenAI",
        sa_column_kwargs={"comment": "Name of the resulting fine-tuned model"},
    )
    train_data_s3_object: str | None = Field(
        default=None,
        description="S3 URL of the training data stored in S3",
        sa_column_kwargs={"comment": "S3 URL of the training data"},
    )
    test_data_s3_object: str | None = Field(
        default=None,
        description="S3 URL of the testing data stored in S3",
        sa_column_kwargs={"comment": "S3 URL of the testing data"},
    )
    error_message: str | None = Field(
        default=None,
        description="Error message for when something failed",
        sa_column_kwargs={"comment": "Error message if the job failed"},
    )
    # Foreign keys
    document_id: UUID = Field(
        foreign_key="document.id",
        nullable=False,
        sa_column_kwargs={"comment": "Reference to the training document"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )

    # Timestamps
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the job was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the job was last updated"},
    )
    deleted_at: datetime | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Timestamp when the job was deleted"},
    )

    # Relationships
    project: Project = Relationship(back_populates="fine_tuning")
    model_evaluation: "ModelEvaluation" = Relationship(back_populates="fine_tuning")


class FineTuningUpdate(SQLModel):
    training_file_id: str | None = None
    train_data_s3_object: str | None = None
    test_data_s3_object: str | None = None
    split_ratio: float | None = None
    provider_job_id: str | None = None
    fine_tuned_model: str | None = None
    status: str | None = None
    error_message: str | None = None


class FineTuningJobPublic(SQLModel):
    """Public response model with job status and metadata."""

    id: int
    split_ratio: float
    base_model: str
    document_id: UUID
    provider_job_id: str | None = None
    train_data_file_url: str | None = None
    test_data_file_url: str | None = None
    status: str
    error_message: str | None = None
    fine_tuned_model: str | None = None
    training_file_id: str | None = None

    inserted_at: datetime
    updated_at: datetime
