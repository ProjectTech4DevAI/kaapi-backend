"""STT Evaluation models for Speech-to-Text evaluation feature."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field as SQLField
from sqlmodel import Relationship, SQLModel

from app.core.util import now

if TYPE_CHECKING:
    from .evaluation import EvaluationDataset, EvaluationRun
    from .organization import Organization
    from .project import Project


class EvaluationType(str, Enum):
    """Type of evaluation dataset/run."""

    TEXT = "text"
    STT = "stt"
    TTS = "tts"


class STTResultStatus(str, Enum):
    """Status of an STT result."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


# Database Models


class STTSample(SQLModel, table=True):
    """Database table for STT audio samples within a dataset."""

    __tablename__ = "stt_sample"

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the STT sample"},
    )

    # Audio file reference
    object_store_url: str = SQLField(
        description="S3 URL of the audio file",
        sa_column_kwargs={"comment": "S3 URL of the audio file"},
    )

    # Language (can be different per sample within a dataset)
    language: str | None = SQLField(
        default=None,
        max_length=10,
        description="ISO 639-1 language code for this sample",
        sa_column_kwargs={"comment": "ISO 639-1 language code for this sample"},
    )

    # Ground truth transcription (optional, for evaluation)
    ground_truth: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Reference transcription for comparison (optional)",
        ),
        description="Reference transcription for comparison",
    )

    # Audio metadata
    duration_seconds: float | None = SQLField(
        default=None,
        description="Audio duration in seconds",
        sa_column_kwargs={"comment": "Audio duration in seconds"},
    )

    sample_metadata: dict[str, Any] | None = SQLField(
        default_factory=dict,
        sa_column=Column(
            JSONB,
            nullable=True,
            comment="Additional metadata (format, bitrate, original filename, etc.)",
        ),
        description="Additional metadata about the audio sample",
    )

    # Foreign keys
    dataset_id: int = SQLField(
        foreign_key="evaluation_dataset.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the parent evaluation dataset"},
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

    # Timestamps
    inserted_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the sample was created"},
    )
    updated_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the sample was last updated"},
    )

    # Relationships
    dataset: "EvaluationDataset" = Relationship()
    organization: "Organization" = Relationship()
    project: "Project" = Relationship()
    results: list["STTResult"] = Relationship(back_populates="sample")


class STTResult(SQLModel, table=True):
    """Database table for STT transcription results."""

    __tablename__ = "stt_result"

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the STT result"},
    )

    # Transcription output
    transcription: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Generated transcription from STT provider",
        ),
        description="Generated transcription from STT provider",
    )

    # Provider info
    provider: str = SQLField(
        max_length=50,
        description="STT provider used (e.g., gemini-2.5-pro)",
        sa_column_kwargs={"comment": "STT provider used (e.g., gemini-2.5-pro)"},
    )

    # Status
    status: str = SQLField(
        default=STTResultStatus.PENDING.value,
        max_length=20,
        description="Result status: pending, completed, failed",
        sa_column_kwargs={"comment": "Result status: pending, completed, failed"},
    )

    # Metrics (null for Phase 1)
    wer: float | None = SQLField(
        default=None,
        description="Word Error Rate (null for Phase 1)",
        sa_column_kwargs={"comment": "Word Error Rate (null for Phase 1)"},
    )
    cer: float | None = SQLField(
        default=None,
        description="Character Error Rate (null for Phase 1)",
        sa_column_kwargs={"comment": "Character Error Rate (null for Phase 1)"},
    )

    # Human feedback
    is_correct: bool | None = SQLField(
        default=None,
        description="Human feedback: transcription correctness",
        sa_column_kwargs={
            "comment": "Human feedback: transcription correctness (null=not reviewed)"
        },
    )
    comment: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Human feedback comment",
        ),
        description="Human feedback comment",
    )

    # Provider response metadata
    provider_metadata: dict[str, Any] | None = SQLField(
        default_factory=dict,
        sa_column=Column(
            JSONB,
            nullable=True,
            comment="Provider-specific response metadata (tokens, latency, etc.)",
        ),
        description="Provider-specific response metadata",
    )

    # Error message if failed
    error_message: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Error message if transcription failed",
        ),
        description="Error message if transcription failed",
    )

    # Foreign keys
    stt_sample_id: int = SQLField(
        foreign_key="stt_sample.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the STT sample"},
    )
    evaluation_run_id: int = SQLField(
        foreign_key="evaluation_run.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the evaluation run"},
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

    # Timestamps
    inserted_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the result was created"},
    )
    updated_at: datetime = SQLField(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the result was last updated"},
    )

    # Relationships
    sample: "STTSample" = Relationship(back_populates="results")
    evaluation_run: "EvaluationRun" = Relationship()
    organization: "Organization" = Relationship()
    project: "Project" = Relationship()


# Pydantic Models for API


class STTSampleCreate(BaseModel):
    """Request model for creating an STT sample."""

    object_store_url: str = Field(..., description="S3 URL of the audio file")
    ground_truth: str | None = Field(
        None, description="Reference transcription (optional)"
    )


class STTSamplePublic(BaseModel):
    """Public model for STT samples."""

    id: int
    object_store_url: str
    language: str | None
    ground_truth: str | None
    duration_seconds: float | None
    sample_metadata: dict[str, Any] | None
    dataset_id: int
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class STTResultPublic(BaseModel):
    """Public model for STT results."""

    id: int
    transcription: str | None
    provider: str
    status: str
    wer: float | None
    cer: float | None
    is_correct: bool | None
    comment: str | None
    provider_metadata: dict[str, Any] | None
    error_message: str | None
    stt_sample_id: int
    evaluation_run_id: int
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class STTResultWithSample(STTResultPublic):
    """STT result with embedded sample data."""

    sample: STTSamplePublic


class STTFeedbackUpdate(BaseModel):
    """Request model for updating human feedback on a result."""

    is_correct: bool | None = Field(None, description="Is the transcription correct?")
    comment: str | None = Field(None, description="Feedback comment")


class STTDatasetCreate(BaseModel):
    """Request model for creating an STT dataset."""

    name: str = Field(..., description="Dataset name", min_length=1)
    description: str | None = Field(None, description="Dataset description")
    language: str | None = Field(None, description="Default language for the dataset")
    samples: list[STTSampleCreate] = Field(
        ..., description="List of audio samples", min_length=1
    )


class STTDatasetPublic(BaseModel):
    """Public model for STT datasets."""

    id: int
    name: str
    description: str | None
    type: str
    language: str | None
    object_store_url: str | None
    dataset_metadata: dict[str, Any]
    sample_count: int = Field(0, description="Number of samples in the dataset")
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class STTDatasetWithSamples(STTDatasetPublic):
    """STT dataset with embedded samples."""

    samples: list[STTSamplePublic]


class STTEvaluationRunCreate(BaseModel):
    """Request model for starting an STT evaluation run."""

    run_name: str = Field(..., description="Name for this evaluation run", min_length=1)
    dataset_id: int = Field(..., description="ID of the STT dataset to evaluate")
    providers: list[str] = Field(
        default=["gemini-2.5-pro"],
        description="List of STT providers to use",
    )
    language: str | None = Field(None, description="Override language for all samples")


class STTEvaluationRunPublic(BaseModel):
    """Public model for STT evaluation runs."""

    id: int
    run_name: str
    dataset_name: str
    type: str
    language: str | None
    providers: list[str] | None
    dataset_id: int
    status: str
    total_items: int
    processed_samples: int
    score: dict[str, Any] | None
    error_message: str | None
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class STTEvaluationRunWithResults(STTEvaluationRunPublic):
    """STT evaluation run with embedded results."""

    results: list[STTResultWithSample]
    results_total: int = Field(0, description="Total number of results")


class AudioUploadResponse(BaseModel):
    """Response model for audio file upload."""

    s3_url: str = Field(..., description="S3 URL of the uploaded audio file")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the audio file")
