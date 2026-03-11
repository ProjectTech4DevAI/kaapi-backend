"""TTS Evaluation models for Text-to-Speech evaluation feature."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel

from app.core.util import now
from app.models.job import JobStatus
from app.services.tts_evaluations.constants import SUPPORTED_TTS_MODELS

if TYPE_CHECKING:
    from app.models import EvaluationDataset, EvaluationRun


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
        sa_column_kwargs={
            "comment": "Timestamp when the result was last updated",
            "onupdate": now,
        },
    )


class TTSSampleCreate(BaseModel):
    """Request model for a single TTS sample."""

    text: str = Field(
        ..., description="Text to synthesize", min_length=1, max_length=5000
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Text must not be empty or whitespace-only")
        return stripped


class TTSDatasetCreate(BaseModel):
    """Request model for creating a TTS dataset."""

    name: str = Field(..., description="Dataset name", min_length=1, max_length=255)
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

    @classmethod
    def from_model(cls, dataset: EvaluationDataset) -> TTSDatasetPublic:
        """Create from an EvaluationDataset model instance."""
        return cls(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            type=dataset.type,
            language_id=dataset.language_id,
            object_store_url=dataset.object_store_url,
            dataset_metadata=dataset.dataset_metadata,
            organization_id=dataset.organization_id,
            project_id=dataset.project_id,
            inserted_at=dataset.inserted_at,
            updated_at=dataset.updated_at,
        )


class TTSResultPublic(BaseModel):
    """Public model for TTS results."""

    id: int
    sample_text: str
    object_store_url: str | None
    signed_url: str | None = None
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

    @classmethod
    def from_model(
        cls,
        result: TTSResult,
        *,
        signed_url: str | None = None,
    ) -> TTSResultPublic:
        """Create from a TTSResult model instance."""
        return cls(
            id=result.id,
            sample_text=result.sample_text,
            object_store_url=result.object_store_url,
            signed_url=signed_url,
            duration_seconds=(result.metadata_ or {}).get("duration_seconds"),
            size_bytes=(result.metadata_ or {}).get("size_bytes"),
            provider=result.provider,
            status=result.status,
            score=result.score,
            is_correct=result.is_correct,
            comment=result.comment,
            error_message=result.error_message,
            evaluation_run_id=result.evaluation_run_id,
            organization_id=result.organization_id,
            project_id=result.project_id,
            inserted_at=result.inserted_at,
            updated_at=result.updated_at,
        )


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
    def validate_models(cls, models: list[str]) -> list[str]:
        """Validate that all models are supported."""
        if not models:
            raise ValueError("At least one model must be specified")
        unsupported = [m for m in models if m not in SUPPORTED_TTS_MODELS]
        if unsupported:
            raise ValueError(
                f"Unsupported model(s): {', '.join(unsupported)}. "
                f"Supported models are: {', '.join(SUPPORTED_TTS_MODELS)}"
            )
        return models


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

    @classmethod
    def from_model(
        cls,
        run: EvaluationRun,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> TTSEvaluationRunPublic:
        """Create from an EvaluationRun model instance."""
        return cls(
            id=run.id,
            run_name=run.run_name,
            dataset_name=run.dataset_name,
            type=run.type,
            language_id=run.language_id,
            models=run.providers,
            dataset_id=run.dataset_id,
            status=run.status,
            total_items=run.total_items,
            score=run.score,
            error_message=run.error_message,
            run_metadata=run_metadata,
            organization_id=run.organization_id,
            project_id=run.project_id,
            inserted_at=run.inserted_at,
            updated_at=run.updated_at,
        )


class TTSEvaluationRunWithResults(TTSEvaluationRunPublic):
    """TTS evaluation run with embedded results."""

    results: list[TTSResultPublic]
    results_total: int = Field(0, description="Total number of results")

    @classmethod
    def from_model(
        cls,
        run: EvaluationRun,
        *,
        results: list[TTSResultPublic] | None = None,
        results_total: int = 0,
        run_metadata: dict[str, Any] | None = None,
    ) -> TTSEvaluationRunWithResults:
        """Create from an EvaluationRun model instance with results."""
        base = TTSEvaluationRunPublic.from_model(run, run_metadata=run_metadata)
        return cls(
            **base.model_dump(),
            results=results or [],
            results_total=results_total,
        )
