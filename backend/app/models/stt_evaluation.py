"""STT Evaluation models for Speech-to-Text evaluation feature."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field as SQLField

from app.models.job import JobStatus
from sqlmodel import SQLModel

from app.core.util import now

# Supported STT models for evaluation
# SUPPORTED_STT_MODELS = ["gemini-2.5-pro", "gemini-2.5-pro", "gemini-2.0-flash"]
SUPPORTED_STT_MODELS = ["gemini-2.5-pro"]


class EvaluationType(str, Enum):
    """Type of evaluation dataset/run."""

    TEXT = "text"
    STT = "stt"
    TTS = "tts"


class STTSample(SQLModel, table=True):
    """Database table for STT audio samples within a dataset."""

    __tablename__ = "stt_sample"

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the STT sample"},
    )

    file_id: int = SQLField(
        foreign_key="file.id",
        nullable=False,
        ondelete="CASCADE",
        description="Reference to the uploaded audio file",
        sa_column_kwargs={
            "comment": "Reference to the uploaded audio file in file table"
        },
    )

    language_id: int | None = SQLField(
        default=None,
        foreign_key="global.languages.id",
        nullable=True,
        description="Reference to the language in the global languages table",
        sa_column_kwargs={"comment": "Foreign key to global.languages table"},
    )

    ground_truth: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Reference transcription for comparison (optional)",
        ),
        description="Reference transcription for comparison",
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


class STTResult(SQLModel, table=True):
    """Database table for STT transcription results."""

    __tablename__ = "stt_result"

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the STT result"},
    )

    transcription: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Generated transcription from STT provider",
        ),
        description="Generated transcription from STT provider",
    )

    provider: str = SQLField(
        max_length=50,
        description="STT provider used (e.g., gemini-2.5-pro)",
        sa_column_kwargs={"comment": "STT provider used (e.g., gemini-2.5-pro)"},
    )

    status: str = SQLField(
        default=JobStatus.PENDING.value,
        max_length=20,
        description="Result status: PENDING, SUCCESS, FAILED",
        sa_column_kwargs={"comment": "Result status: PENDING, SUCCESS, FAILED"},
    )

    score: dict[str, Any] | None = SQLField(
        default=None,
        sa_column=Column(
            JSONB,
            nullable=True,
            comment="Evaluation metrics (e.g., wer, cer, mer, wil) - extensible for future metrics",
        ),
        description="Evaluation metrics such as WER, CER, etc.",
    )

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

    error_message: str | None = SQLField(
        default=None,
        sa_column=Column(
            Text,
            nullable=True,
            comment="Error message if transcription failed",
        ),
        description="Error message if transcription failed",
    )

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


class STTSampleCreate(BaseModel):
    """Request model for creating an STT sample."""

    file_id: int = Field(..., description="ID of the uploaded audio file")
    ground_truth: str | None = Field(
        None, description="Reference transcription (optional)"
    )


class STTSamplePublic(BaseModel):
    """Public model for STT samples."""

    id: int
    file_id: int
    object_store_url: str | None = None  # Populated from file record when needed
    language_id: int | None
    ground_truth: str | None
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
    score: dict[str, Any] | None
    is_correct: bool | None
    comment: str | None
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
    language_id: int | None = Field(
        None, description="ID of the language from global languages table"
    )
    samples: list[STTSampleCreate] = Field(
        ..., description="List of audio samples", min_length=1
    )


class STTDatasetPublic(BaseModel):
    """Public model for STT datasets."""

    id: int
    name: str
    description: str | None
    type: str
    language_id: int | None
    object_store_url: str | None
    dataset_metadata: dict[str, Any]
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
    models: list[str] = Field(
        default_factory=lambda: ["gemini-2.5-pro"],
        description="List of STT models to use",
        min_length=1,
    )

    @field_validator("models")
    @classmethod
    def validate_models(cls, valid_model: list[str]) -> list[str]:
        """Validate that all models are supported."""
        if not valid_model:
            raise ValueError("At least one model must be specified")
        unsupported = [
            models for models in valid_model if models not in SUPPORTED_STT_MODELS
        ]
        if unsupported:
            raise ValueError(
                f"Unsupported model(s): {', '.join(unsupported)}. "
                f"Supported models are: {', '.join(SUPPORTED_STT_MODELS)}"
            )
        return valid_model


class STTEvaluationRunPublic(BaseModel):
    """Public model for STT evaluation runs."""

    id: int
    run_name: str
    dataset_name: str
    type: str
    language_id: int | None
    models: list[str] | None
    dataset_id: int
    status: str
    total_items: int
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

    file_id: int = Field(..., description="ID of the created file record")
    s3_url: str = Field(..., description="S3 URL of the uploaded audio file")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the audio file")
