from pydantic import JsonValue, field_validator, model_validator
from sqlmodel import Field, SQLModel
from typing import Literal

from app.models.llm.constants import TextProvider
from app.models.llm.request import KaapiCompletionConfig, CompletionType, TextLLMParams

# json_schema is validated shallowly at config time: it must be a non-empty
# object-typed dict. Provider strict-mode normalisation is a run-mode concern.
JSON_SCHEMA_OBJECT_TYPE = "object"


class TopicRelevanceFilter(SQLModel):
    """Pre-filter that scores each item's relevance to the assessment topic."""

    prompt: str = Field(
        ...,
        description=(
            "Relevance-scoring prompt. May embed {{col}} placeholders that the "
            "run mode substitutes per dataset row (batch) or pre-fills (response)."
        ),
    )


class DuplicateDetectionFilter(SQLModel):
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


class AssessmentPreFilters(SQLModel):
    """Optional pre-filters applied before the main assessment call."""

    topic_relevance: TopicRelevanceFilter | None = None
    duplicate_detection: DuplicateDetectionFilter | None = None


class AssessmentTextParams(TextLLMParams):
    """Text params + structured-output schema, scoped to assessment."""

    json_schema: dict[str, JsonValue] | None = Field(
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
