from typing import Literal

from pydantic import JsonValue, model_validator
from sqlmodel import Field, SQLModel

from app.models.llm.constants import Provider, TextProvider
from app.models.llm.request import (
    CompletionType,
    KaapiTextCompletionConfig,
    TextLLMParams,
)

# json_output_schema is validated shallowly at config time: it must be a non-empty
# object-typed dict. Provider strict-mode normalisation is a run-mode concern.
JSON_SCHEMA_OBJECT_TYPE = "object"

# Default llm for a pre-filter call when the config does not override it.
DEFAULT_PREFILTER_PROVIDER = Provider.OPENAI
DEFAULT_PREFILTER_MODEL = "gpt-5.6-luna"


class InputColumn(SQLModel):
    """One BATCH input column: its type (required, no default — every column must
    declare one), and how an attachment value is provided (`format`)."""

    type: Literal["text", "image", "pdf"]
    format: Literal["url", "base64"] | None = None


class PreFilterParams(TextLLMParams):
    """Flat, mapper-ready LLM params for a pre-filter call.

    Same knobs as ``TextLLMParams`` (so ``mappers.py`` maps them unchanged),
    including ``instructions`` — a pre-filter carries its criteria in
    ``params.instructions`` exactly like the assessment call. Unknown keys are
    rejected so a mistyped param fails loudly instead of silently no-op'ing.
    """

    model_config = {"extra": "forbid"}

    model: str = Field(default=DEFAULT_PREFILTER_MODEL)
    instructions: str = Field(
        ...,
        min_length=1,
        description="The pre-filter's criteria (system instruction). Mandatory.",
    )


class PreFilterBase(SQLModel):
    """Shared pre-filter fields — each pre-filter runs its own llm call."""

    provider: TextProvider = Field(
        default=DEFAULT_PREFILTER_PROVIDER,
        description="Provider for this pre-filter's llm call.",
    )
    params: dict[str, JsonValue] = Field(
        default_factory=lambda: {"model": DEFAULT_PREFILTER_MODEL},
        description="PreFilterParams for this pre-filter's llm call (model, temperature, ...).",
    )

    @model_validator(mode="after")
    def _validate_prefilter_params(self):
        params = PreFilterParams.model_validate(self.params).model_dump(
            exclude_none=True
        )
        # if default pre-filter model is used, then set the default effort and summary params, and remove temperature (which is not used for pre-filtering)
        if params.get("model") == DEFAULT_PREFILTER_MODEL:
            params.setdefault("effort", "high")
            params.setdefault("summary", "auto")
            params.pop("temperature", None)
        self.params = params
        return self


class TopicRelevanceFilter(PreFilterBase):
    """Pre-filter that scores each item's relevance to the assessment topic.

    Its criteria live in ``params.instructions`` (like the assessment call); the
    item's own columns + attachments are the user content.
    """

    stop_on_fail: bool = Field(
        default=True,
        description=(
            "If true, a failing verdict stops the chain and skips the assessment for "
            "that item; if false, the verdict is just recorded and the assessment runs."
        ),
    )


class DuplicateDetectionFilter(PreFilterBase):
    """Pre-filter that flags items duplicating prior corpus content.

    Its criteria live in ``params.instructions`` (like the assessment call).
    """

    knowledge_base_id: str | None = Field(
        default=None,
        description="Vector store to compare against; defaults to the platform corpus when unset.",
    )
    stop_on_fail: bool = Field(
        default=False,
        description=(
            "If true, a failing verdict stops the chain and skips the assessment for "
            "that item; if false, the verdict is just recorded and the assessment runs."
        ),
    )


class AssessmentPreFilters(SQLModel):
    """Optional pre-filters applied before the main assessment call."""

    topic_relevance: TopicRelevanceFilter | None = None
    duplicate_detection: DuplicateDetectionFilter | None = None


class AssessmentTextParams(TextLLMParams):
    """Text params + structured input/output schemas, scoped to assessment."""

    input_schema: dict[str, InputColumn] = Field(
        ...,
        min_length=1,
        description=(
            "Per-column spec for BATCH submissions ({type, format}). Mandatory and "
            "non-empty; every declared column must be present in every submission row."
        ),
    )
    json_output_schema: dict[str, JsonValue] | None = Field(
        default=None,
        description="Object-typed JSON schema for structured output. Omit for free-form text.",
    )


class AssessmentCompletionConfig(KaapiTextCompletionConfig):
    provider: TextProvider = Field(
        ..., description="Provider to use for the assessment completion call."
    )
    type: Literal[CompletionType.TEXT] = CompletionType.TEXT
    # Overrides the inherited `TextLLMParams` field with the assessment-scoped
    # superset so input_schema/json_output_schema survive field validation
    # instead of being dropped as unknown TextLLMParams keys.
    params: AssessmentTextParams = Field(
        ...,
        description="Assessment-scoped Kaapi text params (adds input/output schemas).",
    )

    @model_validator(mode="after")
    def validate_params(self):
        # SQLModel constructs nested non-table submodels twice for a field this
        # validator mutates (once via the nested model's own __init__, once via
        # the outer model's compiled validator) — the second pass sees the dict
        # this validator already produced, so treat that as a no-op.
        if isinstance(self.params, dict):
            return self
        user_set_temp = "temperature" in self.params.model_fields_set
        dumped = self.params.model_dump(exclude_none=True)
        if not user_set_temp:
            dumped.pop("temperature", None)
        self.params = dumped
        return self


class AssessmentConfigBlob(SQLModel):
    """config_blob shape for a config tagged ASSESSMENT.

    De-associated from ConfigBlob: no inference_mode (the run mode is chosen by
    the caller, not the config), no columns/attachment_columns (mode-agnosticism
    comes from {{col}} prompt interpolation), and no post_processing.
    """

    pre_filters: AssessmentPreFilters | None = None
    assessment: AssessmentCompletionConfig
