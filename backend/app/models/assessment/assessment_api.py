"""API-client request/response models for ``POST /assessments``.

Method (RESPONSE vs BATCH) inferred from input shape. Shared enums, tables, and legacy
RUN models live in ``assessment.py``.
"""

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from sqlmodel import SQLModel

from app.models.assessment.assessment import (
    AssessmentConfigRef,
    AssessmentMethod,
    AssessmentStatus,
)
from app.models.llm.request import ImageInput, PDFInput
from app.models.llm.response import Usage

Attachment = Annotated[ImageInput | PDFInput, Field(discriminator="type")]


class ResponseInput(SQLModel):
    """RESPONSE input — a single `query` + optional attachments (no columns)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="User query")
    attachments: list[Attachment] = Field(default_factory=list)


# One BATCH submission row: column -> string value (text, or a url/base64 value for
# attachment columns). Column type/strict/format live in the config's input_schema.
Submission = dict[str, str]


class BatchInput(SQLModel):
    """BATCH input — a `query` template + a list of submission rows."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ..., min_length=1, description="Query template with {column} placeholders"
    )
    data: list[Submission] = Field(
        ..., min_length=1, description="Submission rows; one assessed item each"
    )


# Strict, tagless discrimination via extra=forbid: an input carrying `data` is a
# BatchInput, one carrying `attachments` (or query-only) is a ResponseInput. The two
# cannot be confused — neither accepts the other's distinguishing field.
AssessmentInput = ResponseInput | BatchInput


def derive_method(
    input_: AssessmentInput | None, dataset_id: int | None
) -> AssessmentMethod:
    """Infer method: ResponseInput ⇒ RESPONSE, BatchInput ⇒ BATCH, else dataset_id ⇒ RUN."""
    if isinstance(input_, ResponseInput):
        return AssessmentMethod.RESPONSE
    if isinstance(input_, BatchInput):
        return AssessmentMethod.BATCH
    if dataset_id is not None:
        return AssessmentMethod.RUN
    raise ValueError("[derive_method] Provide inline `input` or `dataset_id`")


class AssessmentCreate(BaseModel):
    """New API create request; one config, method inferred from the input type."""

    config: AssessmentConfigRef
    input: AssessmentInput
    callback_url: HttpUrl = Field(
        ..., description="Webhook the result is POSTed to on completion (required)"
    )
    request_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Passed through unchanged in the callback for correlation",
    )


class AssessmentSubmitResponse(BaseModel):
    """Flat API-client submit ack (one config, so one execution) — mirrors the llm-call ack."""

    assessment_id: UUID
    status: AssessmentStatus
    message: str
    inserted_at: datetime
    updated_at: datetime


class PreFilterVerdict(BaseModel):
    """Structured pre-filter result per item."""

    verdict: bool
    reasoning: str = ""


class PreFilter(BaseModel):
    """Grouped pre-filter verdicts for one item; each is null if not configured."""

    topic_relevance: PreFilterVerdict | None = None
    duplicate_detection: PreFilterVerdict | None = None


class AssessmentOutput(BaseModel):
    """Per-item output: parsed assessment plus grouped pre-filter verdicts.

    ``assessment`` is a dict when the config emits a structured (json_output_schema)
    output, a raw string for free-text output, or null for gated/failed rows with no result.
    """

    assessment: dict[str, Any] | str | None = None
    pre_filter: PreFilter | None = None


class AssessmentMeta(BaseModel):
    """Provider-call metadata for one assessed item; null when no assessment call ran."""

    provider: str
    model: str
    response_id: str | None = None
    usage: Usage | None = None


class AssessmentResult(BaseModel):
    """Shared result — one BATCH item or the single RESPONSE result."""

    output: AssessmentOutput
    metadata: AssessmentMeta | None = None
    error: str | None = None


class AssessmentCounts(BaseModel):
    """Per-execution tallies: assessed rows, gate-filtered rows, errored rows."""

    assessed: int = 0
    filtered: int = 0
    errors: int = 0


class AssessmentBatchResult(BaseModel):
    """BATCH result body: counts + one AssessmentResult per input row."""

    total_items: int
    counts: AssessmentCounts = AssessmentCounts()
    items: list[AssessmentResult] = []


# The `data` body of a response, keyed by inference method: a single AssessmentResult
# (RESPONSE) or an AssessmentBatchResult carrying the per-row list (BATCH).
AssessmentResultData = AssessmentResult | AssessmentBatchResult


class AssessmentCallback(BaseModel):
    """Webhook payload: the same skeleton as the status response plus the echoed request_metadata."""

    assessment_id: UUID
    status: AssessmentStatus
    data: AssessmentResultData | None = None
    request_metadata: dict[str, Any] | None = None
