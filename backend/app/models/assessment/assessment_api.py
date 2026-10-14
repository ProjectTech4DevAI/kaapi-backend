"""API-client request/response models for ``POST /assessments``.

Method (RESPONSE vs BATCH) inferred from input shape. Shared enums, tables, and the
UI-only RUN models live in ``assessment.py``.
"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, JsonValue, model_validator
from sqlmodel import SQLModel

from app.models.assessment.assessment import (
    AssessmentConfigRef,
    AssessmentMethod,
    AssessmentStatus,
    StageStatus,
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


class ApiStage(StrEnum):
    """Pipeline stage identifiers; names match the pre-filter config + result fields."""

    TOPIC_RELEVANCE = "topic_relevance"
    ASSESSMENT = "assessment"


class StageKind(StrEnum):
    GATE = "GATE"
    PASS_THROUGH = "PASS_THROUGH"
    ASSESSMENT = "ASSESSMENT"


class PreFilterVerdict(BaseModel):
    """Structured pre-filter result per item."""

    verdict: bool
    reasoning: str = ""


class PipelineStep(BaseModel):
    stage: ApiStage
    kind: StageKind


class StageCounters(BaseModel):
    total: int = 0
    passed: int = 0
    rejected: int = 0


class AssessmentExecution(BaseModel):
    """Runtime state of the staged BATCH pipeline, stored on ``assessment.execution``.

    Advanced one Celery task at a time by ``run_batch_stage``; ``stage_status`` keys the
    idempotent redelivery. Deliberately O(stages), never O(rows): per-row verdicts and
    outputs live in each stage's dump (``assessment.result_files``), not here.
    """

    model_config = ConfigDict(extra="forbid")

    pipeline: list[PipelineStep]
    stage: ApiStage
    stage_status: StageStatus
    stage_batches: dict[ApiStage, int] = Field(default_factory=dict)
    stage_errors: dict[ApiStage, dict[int, str]] = Field(default_factory=dict)
    counters: dict[ApiStage, StageCounters] = Field(default_factory=dict)
    callback_url: str | None = None
    request_metadata: dict[str, JsonValue] | None = None
    error: str | None = None

    def stage_kind(self, stage: ApiStage) -> StageKind:
        for step in self.pipeline:
            if step.stage == stage:
                return step.kind
        raise ValueError(f"[stage_kind] Stage {stage} not in pipeline")

    def next_stage(self, current: ApiStage) -> ApiStage | None:
        stages = [step.stage for step in self.pipeline]
        idx = stages.index(current)
        return stages[idx + 1] if idx + 1 < len(stages) else None

    def gate_stages(self) -> list[ApiStage]:
        return [step.stage for step in self.pipeline if step.kind == StageKind.GATE]


class ParsedResult(TypedDict):
    """One provider batch response row, normalised across providers.

    ``usage`` stays an open dict — the token-count keys differ per provider
    (``input_tokens`` vs ``prompt_tokens``, ...)."""

    output: str | None
    error: str | None
    usage: dict[str, Any] | None
    response_id: str | None


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
    experiment_name: str | None = Field(
        default=None, description="Run label shown in the console"
    )
    callback_url: HttpUrl | None = Field(
        default=None,
        description=(
            "Webhook the result is POSTed to on completion; omit to poll "
            "GET /assessments/{assessment_id} instead"
        ),
    )
    request_metadata: dict[str, JsonValue] | None = Field(
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


class AssessmentResultRow(AssessmentResult):
    """One row as the poll endpoint returns it: the webhook item plus its origin.

    ``output`` is byte-identical to the webhook's, so one client parser handles both.
    ``input`` is null when the stored submission rows could not be read.
    """

    row_index: int
    input: Submission | None = None


class AssessmentSummary(BaseModel):
    """One row of the assessment list: everything readable without touching storage.

    Deliberately carries no per-row counts — those need the provider dump streamed
    back, which a list (and a poll on it) must not pay for. Use the detail endpoint.
    """

    assessment_id: UUID
    method: AssessmentMethod
    status: AssessmentStatus
    experiment_name: str | None = None
    submission_id: UUID | None = None
    submission_name: str | None = None
    config: AssessmentConfigRef | None = None
    total_items: int = 0
    # This run's own stages, in order — a run with no pre-filter has one entry.
    stages: list[str] = []
    stage: str | None = None
    stage_status: str | None = None
    error: str | None = None
    inserted_at: datetime
    updated_at: datetime


class AssessmentDetailResponse(AssessmentSummary):
    """Poll response: the summary plus every row produced so far.

    Safe to read mid-run — rows the provider has not returned yet carry
    ``output.assessment = null``.
    """

    counts: AssessmentCounts = AssessmentCounts()
    items: list[AssessmentResultRow] = []


# The `data` body of a response, keyed by inference method: a single AssessmentResult
# (RESPONSE) or an AssessmentBatchResult carrying the per-row list (BATCH).
AssessmentResultData = AssessmentResult | AssessmentBatchResult


class AssessmentCallback(BaseModel):
    """Webhook payload: the same skeleton as the status response plus the echoed request_metadata."""

    assessment_id: UUID
    status: AssessmentStatus
    data: AssessmentResultData | None = None
    request_metadata: dict[str, JsonValue] | None = None


class AssessmentResultFiles(BaseModel):
    """URLs of the provider dumps for each stage, if any."""

    topic_relevance: str | None = None
    assessment: str | None = None
    errors: str | None = None
