import logging
import re
from typing import Literal

from pydantic import JsonValue, model_validator
from sqlmodel import Field, SQLModel

from app.models.llm.constants import Provider, TextProvider
from app.models.llm.request import CompletionType, KaapiCompletionConfig, TextLLMParams

logger = logging.getLogger(__name__)

# json_output_schema is validated shallowly at config time: it must be a non-empty
# object-typed dict. Provider strict-mode normalisation is a run-mode concern.
JSON_SCHEMA_OBJECT_TYPE = "object"

# {column} placeholders in a submission template; the capture group is the column name.
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

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
    submission: str | None = Field(
        default=None,
        description=(
            "Optional per-row prompt template with {column} placeholders for this "
            "pre-filter. Placeholders must resolve against the blob-level input_schema."
        ),
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


class AssessmentPreFilters(SQLModel):
    """Optional pre-filters applied before the main assessment call."""

    topic_relevance: TopicRelevanceFilter | None = None


class AssessmentTextParams(TextLLMParams):
    """Text params + structured output schema, scoped to assessment."""

    submission: str = Field(
        ...,
        min_length=1,
        description=(
            "Per-row prompt template with {column} placeholders. Mandatory; every "
            "placeholder must resolve against input_schema."
        ),
    )
    json_output_schema: dict[str, JsonValue] | None = Field(
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

    input_schema: dict[str, InputColumn] = Field(
        ...,
        min_length=1,
        description=(
            "Per-column spec for the BATCH `data` rows ({type, format}). Shared by the "
            "pre-filter and assessment consumers, so it lives once at the blob root. "
            "Mandatory and non-empty; every declared column must be present in every row."
        ),
    )
    pre_filters: AssessmentPreFilters | None = None
    assessment: AssessmentCompletionConfig

    @model_validator(mode="after")
    def _validate_submission_placeholders(self):
        """Every {column} in every submission template must be a declared input_schema key.

        Pre-filter and assessment submissions both validate against the blob-level
        input_schema. Each params dict is already plain (their own validators dumped the
        typed params), so submissions are read via dict access.
        """
        declared = set(self.input_schema)

        blocks: list[tuple[str, str]] = [
            ("assessment", self.assessment.params.get("submission") or "")
        ]
        topic_relevance = self.pre_filters.topic_relevance if self.pre_filters else None
        if topic_relevance is not None and topic_relevance.params.get("submission"):
            blocks.append(
                ("pre_filter topic_relevance", str(topic_relevance.params["submission"]))
            )

        for block, template in blocks:
            unknown = sorted(set(PLACEHOLDER_RE.findall(template)) - declared)
            if unknown:
                logger.error(
                    f"[_validate_submission_placeholders] Unknown placeholder(s) "
                    f"| block: {block} | unknown: {unknown} | declared: {sorted(declared)}"
                )
                raise ValueError(
                    f"{block} submission references unknown input_schema column(s): "
                    f"{', '.join(unknown)}"
                )
        return self
