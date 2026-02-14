"""TTS Evaluation models for Text-to-Speech evaluation feature."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel

from app.core.util import now
from app.models.job import JobStatus
from app.models.stt_evaluation import EvaluationType

# Supported TTS models for evaluation
SUPPORTED_TTS_MODELS = ["gemini-2.5-pro-preview-tts"]


class TTSResult(SQLModel, table=True):
    """Database table for TTS synthesis results."""

    __tablename__ = "tts_result"

    id: int = SQLField(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the TTS result"},
    )

    sample_text: str = SQLField(
        sa_column=Column(
            Text,
            nullable=False,
            comment="Input text that was synthesized to speech",
        ),
        description="Input text that was synthesized to speech",
    )

    object_store_url: str | None = SQLField(
        default=None,
        sa_column_kwargs={"comment": "S3 URL of the generated WAV audio file"},
    )

    metadata_: dict[str, Any] | None = SQLField(
        default=None,
        sa_column=Column(
            "metadata",
            JSONB,
            nullable=True,
            comment="Audio metadata: {duration_seconds, size_bytes}",
        ),
        description="Audio metadata (duration_seconds, size_bytes)",
    )

    provider: str = SQLField(
        max_length=100,
        description="TTS provider used (e.g., gemini-2.5-pro-preview-tts)",
        sa_column_kwargs={
            "comment": "TTS provider used (e.g., gemini-2.5-pro-preview-tts)"
        },
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
            comment="Extensible evaluation metrics (null in Phase 1)",
        ),
        description="Extensible evaluation metrics",
    )

    is_correct: bool | None = SQLField(
        default=None,
        description="Human feedback: audio quality correctness",
        sa_column_kwargs={
            "comment": "Human feedback: audio quality correctness (null=not reviewed)"
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
            comment="Error message if synthesis failed",
        ),
        description="Error message if synthesis failed",
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


# --- Pydantic request/response models ---


class TTSSampleCreate(BaseModel):
    """Request model for a single TTS sample."""

    text: str = Field(..., description="Text to synthesize", min_length=1)


class TTSDatasetCreate(BaseModel):
    """Request model for creating a TTS dataset."""

    name: str = Field(..., description="Dataset name", min_length=1)
    description: str | None = Field(None, description="Dataset description")
    language_id: int | None = Field(
        None, description="ID of the language from global languages table"
    )
    samples: list[TTSSampleCreate] = Field(
        ..., description="List of text samples", min_length=1
    )


class TTSDatasetPublic(BaseModel):
    """Public model for TTS datasets."""

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


class TTSResultPublic(BaseModel):
    """Public model for TTS results."""

    id: int
    sample_text: str
    object_store_url: str | None
    duration_seconds: float | None = None
    size_bytes: int | None = None
    provider: str
    status: str
    score: dict[str, Any] | None
    is_correct: bool | None
    comment: str | None
    error_message: str | None
    evaluation_run_id: int
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class TTSFeedbackUpdate(BaseModel):
    """Request model for updating human feedback on a TTS result."""

    is_correct: bool | None = Field(
        None, description="Is the synthesized audio correct?"
    )
    comment: str | None = Field(None, description="Feedback comment")


class TTSEvaluationRunCreate(BaseModel):
    """Request model for starting a TTS evaluation run."""

    run_name: str = Field(..., description="Name for this evaluation run", min_length=1)
    dataset_id: int = Field(..., description="ID of the TTS dataset to evaluate")
    models: list[str] = Field(
        default_factory=lambda: ["gemini-2.5-pro-preview-tts"],
        description="List of TTS models to use",
        min_length=1,
    )

    @field_validator("models")
    @classmethod
    def validate_models(cls, valid_model: list[str]) -> list[str]:
        """Validate that all models are supported."""
        if not valid_model:
            raise ValueError("At least one model must be specified")
        unsupported = [m for m in valid_model if m not in SUPPORTED_TTS_MODELS]
        if unsupported:
            raise ValueError(
                f"Unsupported model(s): {', '.join(unsupported)}. "
                f"Supported models are: {', '.join(SUPPORTED_TTS_MODELS)}"
            )
        return valid_model


class TTSEvaluationRunPublic(BaseModel):
    """Public model for TTS evaluation runs."""

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
    run_metadata: dict[str, Any] | None = None
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class TTSEvaluationRunWithResults(TTSEvaluationRunPublic):
    """TTS evaluation run with embedded results."""

    results: list[TTSResultPublic]
    results_total: int = Field(0, description="Total number of results")
