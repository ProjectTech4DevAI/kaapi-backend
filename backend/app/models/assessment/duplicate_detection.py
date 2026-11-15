from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, JsonValue

from app.models.assessment.assessment import AssessmentConfigRef, AssessmentStatus

MIN_SIMILARITY_SCORE = 1
MAX_SIMILARITY_SCORE = 5


class DetectionLevel(StrEnum):
    """The two checks of a job; also marks which run row belongs to which check."""

    STRICT = "strict"
    LENIENT = "lenient"


class StrictVerdict(BaseModel):
    """Strict result: whether the query duplicates a corpus entry, and that entry's text."""

    is_duplicate: bool
    citation: str | None = None


class LenientMatch(BaseModel):
    """One similar corpus entry and how likely it is a duplicate of the query."""

    score: int = Field(ge=MIN_SIMILARITY_SCORE, le=MAX_SIMILARITY_SCORE)
    citation: str


class DetectionErrors(BaseModel):
    """Why a level produced no result; null for a level that succeeded."""

    strict: str | None = None
    lenient: str | None = None


class DuplicateDetectionResult(BaseModel):
    """Both levels' results; a level that failed is null here and explained in errors."""

    strict: StrictVerdict | None = None
    lenient: list[LenientMatch] | None = None
    errors: DetectionErrors = DetectionErrors()


class OutputSchema(BaseModel):
    """The parts of a config's JSON output schema a check looks at: declared fields and array items."""

    properties: dict[str, "OutputSchema"] = {}
    items: "OutputSchema | None" = None


class DetectionParams(BaseModel):
    """What a check reads from its config's assessment params; anything else is the user's."""

    model: str
    knowledge_base_ids: list[str] = Field(min_length=1)
    max_num_results: int = Field(ge=1)
    json_output_schema: OutputSchema


class DuplicateDetectionConfig(BaseModel):
    """The saved ASSESSMENT config version each check runs with."""

    strict: AssessmentConfigRef
    lenient: AssessmentConfigRef


class DuplicateDetectionCreate(BaseModel):
    """Request to check one submission against the strict and lenient corpora."""

    query: str = Field(min_length=1, description="Submission text to check")
    config: DuplicateDetectionConfig
    callback_url: HttpUrl = Field(description="Webhook the result is POSTed to")
    request_metadata: dict[str, JsonValue] | None = Field(
        default=None,
        description="Passed through unchanged in the callback for correlation",
    )


class DuplicateDetectionPublic(DuplicateDetectionResult):
    """A job as returned on submit and on status: its state plus any results so far."""

    id: UUID
    status: AssessmentStatus
    config: DuplicateDetectionConfig
    request_metadata: dict[str, JsonValue] | None = None
    inserted_at: datetime
    updated_at: datetime
