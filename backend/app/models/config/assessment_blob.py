from pydantic import JsonValue, field_validator, model_validator
from sqlmodel import Field, SQLModel

from app.models.llm.constants import CompletionType
from app.models.llm.request import CompletionConfig

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


class AssessmentBlock(SQLModel):
    """The core assessment call: system prompt, output schema, and model."""

    system_prompt: str = Field(
        ...,
        description=(
            "System prompt for the assessment. May embed {{col}} placeholders "
            "resolved by the run mode."
        ),
    )
    json_schema: dict[str, JsonValue] = Field(
        ...,
        description="Object-typed JSON schema describing the structured assessment output.",
    )
    model: CompletionConfig = Field(
        ..., description="Shared LLM completion config used to run the assessment."
    )

    @model_validator(mode="before")
    @classmethod
    def _lift_flat_model_config(cls, data: object) -> object:
        """Accept the contract's flat model shape and lift it into a CompletionConfig.

        The contract sends `{provider, model, temperature, max_output_tokens, ...}`;
        the stored/validated type is the shared CompletionConfig union, whose Kaapi
        variant nests those params under `params`. A payload that already carries
        `type`/`params` is passed through untouched.
        """
        if not isinstance(data, dict):
            return data
        model = data.get("model")
        if not isinstance(model, dict) or "type" in model or "params" in model:
            return data
        data["model"] = {
            "provider": model.get("provider"),
            "type": CompletionType.TEXT.value,
            "params": {k: v for k, v in model.items() if k != "provider"},
        }
        return data

    @field_validator("json_schema")
    @classmethod
    def _validate_json_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if not value:
            raise ValueError("json_schema must be a non-empty object")
        if value.get("type") != JSON_SCHEMA_OBJECT_TYPE:
            raise ValueError(f"json_schema.type must be '{JSON_SCHEMA_OBJECT_TYPE}'")
        return value


class AssessmentConfigBlob(SQLModel):
    """config_blob shape for a config tagged ASSESSMENT.

    De-associated from ConfigBlob: no inference_mode (the run mode is chosen by
    the caller, not the config), no columns/attachment_columns (mode-agnosticism
    comes from {{col}} prompt interpolation), and no post_processing.
    """

    pre_filters: AssessmentPreFilters | None = None
    assessment: AssessmentBlock
