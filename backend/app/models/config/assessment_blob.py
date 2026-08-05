from typing import Literal

from pydantic import JsonValue, model_validator
from sqlmodel import Field, SQLModel

from app.models.llm.constants import Provider, TextProvider
from app.models.llm.request import CompletionType, KaapiCompletionConfig, TextLLMParams

# json_schema_output is validated shallowly at config time: it must be a non-empty
# object-typed dict. Provider strict-mode normalisation is a run-mode concern.
JSON_SCHEMA_OBJECT_TYPE = "object"

# Default llm for a pre-filter call when the config does not override it.
DEFAULT_PREFILTER_PROVIDER = Provider.OPENAI
DEFAULT_PREFILTER_MODEL = "gpt-5.6-luna"


class InputColumn(SQLModel):
    """One BATCH input column: its type, whether it must be present in every
    submission (`strict`), and how an attachment value is provided (`format`)."""

    type: Literal["text", "image", "pdf"] = "text"
    strict: bool = False
    format: Literal["url", "base64"] | None = None


class PreFilterBase(SQLModel):
    """Shared pre-filter fields — each pre-filter runs its own llm call."""

    provider: TextProvider = Field(
        default=DEFAULT_PREFILTER_PROVIDER,
        description="Provider for this pre-filter's llm call.",
    )
    model: str = Field(
        default=DEFAULT_PREFILTER_MODEL,
        description="Model for this pre-filter's llm call.",
    )


class TopicRelevanceFilter(PreFilterBase):
    """Pre-filter that scores each item's relevance to the assessment topic."""

    prompt: str = Field(
        ...,
        description=(
            "Relevance-scoring prompt. May embed {{col}} placeholders that the "
            "run mode substitutes per dataset row (batch) or pre-fills (response)."
        ),
    )
    stop_on_fail: bool = Field(
        default=True,
        description=(
            "If true, a failing verdict stops the chain and skips the assessment for "
            "that item; if false, the verdict is just recorded and the assessment runs."
        ),
    )


class DuplicateDetectionFilter(PreFilterBase):
    """Pre-filter that flags items duplicating prior corpus content."""

    content: str | None = Field(
        default=None,
        description=(
            "Duplicate-comparison template. May embed {{col}} placeholders "
            "resolved by the run mode."
        ),
    )
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

    input_schema: dict[str, InputColumn] | None = Field(
        default=None,
        description=(
            "Per-column spec for BATCH submissions ({type, strict, format}). A column "
            "marked strict must be present in every submission."
        ),
    )
    json_schema_output: dict[str, JsonValue] | None = Field(
        default=None,
        description="Object-typed JSON schema for structured output. Omit for free-form text.",
    )


class AssessmentCompletionConfig(KaapiCompletionConfig):
    provider: TextProvider = Field(
        ..., description="Provider to use for the assessment completion call."
    )
    type: Literal[CompletionType.TEXT] = CompletionType.TEXT

    @model_validator(mode="after")
    def validate_params(self):  # overrides KaapiCompletionConfig.validate_params
        user_set_temp = "temperature" in self.params
        validated = AssessmentTextParams.model_validate(self.params)
        self.params = validated.model_dump(exclude_none=True)
        if not user_set_temp:
            self.params.pop("temperature", None)
        return self


class AssessmentConfigBlob(SQLModel):
    """config_blob shape for a config tagged ASSESSMENT.

    De-associated from ConfigBlob: no inference_mode (the run mode is chosen by
    the caller, not the config), no columns/attachment_columns (mode-agnosticism
    comes from {{col}} prompt interpolation), and no post_processing.
    """

    pre_filters: AssessmentPreFilters | None = None
    assessment: AssessmentCompletionConfig
