"""API-client request/response models for ``POST /assessments``.

Method (RESPONSE vs BATCH) inferred from input shape. Shared enums, tables, and legacy
RUN models live in ``assessment.py``.
"""

from datetime import datetime
from typing import Annotated, Any, NotRequired, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator
from sqlmodel import SQLModel

from app.models.assessment.assessment import (
    AssessmentConfigRef,
    AssessmentMethod,
    AssessmentStatus,
)
from app.models.llm.request import ImageInput, PDFInput

Attachment = Annotated[ImageInput | PDFInput, Field(discriminator="type")]


class ResponseInput(SQLModel):
    """RESPONSE input — optional attachments (no columns). The prompt lives in the config."""

    model_config = ConfigDict(extra="forbid")

    attachments: list[Attachment] = Field(default_factory=list)


# One BATCH submission row: column -> string value (text, or a url/base64 value for
# attachment columns). Column type/strict/format live in the config's input_schema.
Submission = dict[str, str]


class BatchInput(SQLModel):
    """BATCH input — rows inline, or a pointer to an uploaded submission file.

    The two are mutually exclusive: exactly one must be given. The prompt template
    lives in the config either way.
    """

    model_config = ConfigDict(extra="forbid")

    data: list[Submission] | None = Field(
        default=None,
        min_length=1,
        description="Submission rows; one assessed item each",
    )
    submission_doc_id: UUID | None = Field(
        default=None,
        description="Id of an uploaded submission file to read the rows from",
    )

    @model_validator(mode="after")
    def _exactly_one_row_source(self) -> "BatchInput":
        if (self.data is None) == (self.submission_doc_id is None):
            raise ValueError(
                "Provide exactly one of 'data' (inline rows) or 'submission_doc_id'."
            )
        return self


class Verdict(TypedDict):
    """A pre-filter's parsed judgement for one row."""

    verdict: bool
    reasoning: str


class ParsedResult(TypedDict):
    """One provider batch response row, normalised across providers.

    ``usage`` stays an open dict — the token-count keys differ per provider
    (``input_tokens`` vs ``prompt_tokens``, ...)."""

    output: str | None
    error: str | None
    usage: dict[str, Any] | None
    response_id: str | None


class BatchRunState(TypedDict):
    """Persisted runtime state of the staged BATCH pipeline, stored on
    ``AssessmentRun.execution`` (JSONB). Advanced one Celery tick at a time by
    ``run_batch_stage``; keyed off ``stage_status`` for idempotent redelivery."""

    pipeline: list[dict[str, str]]  # ordered [{"stage","kind"}]
    stage: str  # current stage
    stage_status: str  # PENDING | PROCESSING | COMPLETED | FAILED
    # Values are None-typed at the write sites: batch_job.id is an ORM-optional PK and
    # raw_output_url is Optional, so the map value types must admit None.
    stage_batches: dict[str, int | None]  # stage -> provider batch_job id
    stage_output_urls: dict[str, str | None]  # stage -> raw result url
    # stage -> {row_index -> error}, captured at parse time (raw dumps are too big here)
    stage_errors: NotRequired[dict[str, dict[str, str]]]
    verdicts: dict[str, dict[str, Verdict]]  # stage -> {item_idx -> verdict}
    counters: dict[str, dict[str, int]]  # stage -> {total,passed,rejected}
    gate_passed: list[bool]  # per-item still-eligible flag
    provider: str
    model: str
    input_schema: dict[str, Any] | None
    callback_url: str
    request_metadata: dict[str, Any] | None
    error: NotRequired[str]  # only set on failure (_fail)


# Strict, tagless discrimination via extra=forbid: an input carrying `data` is a
# BatchInput, one carrying only `attachments` is a ResponseInput. The two cannot be
# confused — neither accepts the other's distinguishing field.
AssessmentInput = ResponseInput | BatchInput


def derive_method(
    input_: AssessmentInput | None, submission_id: UUID | None
) -> AssessmentMethod:
    """Infer method: ResponseInput ⇒ RESPONSE, BatchInput ⇒ BATCH, else submission_id ⇒ RUN."""
    if isinstance(input_, ResponseInput):
        return AssessmentMethod.RESPONSE
    if isinstance(input_, BatchInput):
        return AssessmentMethod.BATCH
    if submission_id is not None:
        return AssessmentMethod.RUN
    raise ValueError("[derive_method] Provide inline `input` or `submission_id`")


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


class AssessmentOutput(BaseModel):
    """Per-item output: parsed assessment plus grouped pre-filter verdicts.

    ``assessment`` is a dict when the config emits a structured (json_output_schema)
    output, a raw string for free-text output, or null for gated/failed rows with no result.
    """

    assessment: dict[str, Any] | str | None = None
    pre_filter: PreFilter | None = None


class AssessmentResult(BaseModel):
    """Shared result — one BATCH item or the single RESPONSE result."""

    output: AssessmentOutput
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
