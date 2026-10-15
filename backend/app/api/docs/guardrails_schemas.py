"""JSON Schemas and examples for the guardrails management-API proxy routes.

These routes forward to the kaapi-guardrails service, which owns validation.
Describing the contract as data here (instead of as Pydantic models the
gateway would enforce) keeps Swagger fully typed and testable while letting
every payload reach upstream: a validator type or field this gateway has not
heard of yet is forwarded, not rejected.

Keep in sync with `kaapi-guardrails/backend/app/schemas/` on a best-effort
basis — a stale description misleads, but a stale schema no longer blocks a
working call.
"""

import json
from typing import Any, NamedTuple

from app.api.openapi_docs import ResponseDoc, ref
from app.services.llm.guardrails import EXPOSED_VALIDATOR_TYPES

_UUID_EXAMPLE = "0b9f1a4c-2d3e-4f50-8a6b-7c8d9e0f1a2b"
_TIMESTAMP_EXAMPLE = "2026-10-14T09:30:00"


def _envelope(data_schema: dict[str, Any]) -> dict[str, Any]:
    """The `{success, data, error, metadata}` wrapper the service returns."""
    return {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "data": data_schema,
            "error": {"type": "string", "nullable": True},
            "metadata": {"type": "object", "nullable": True},
        },
        "required": ["success"],
    }


def _list_envelope(item_schema: dict[str, Any]) -> dict[str, Any]:
    return _envelope({"type": "array", "items": item_schema})


_BAN_LIST_FIELDS: dict[str, Any] = {
    "name": {
        "type": "string",
        "description": "Human-readable name for the ban list. Must be unique.",
    },
    "description": {
        "type": "string",
        "description": "What this ban list covers and when it should be applied.",
    },
    "banned_words": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Words to redact. Consumed by the `ban_list` validator when it is "
            "given this list's `ban_list_id`."
        ),
    },
    "domain": {
        "type": "string",
        "description": (
            "Caller-defined grouping label (e.g. 'abuse'). Used to filter "
            "lists on GET /guardrails/ban_lists."
        ),
    },
    "is_public": {
        "type": "boolean",
        "default": False,
        "description": (
            "When true the list is readable by other tenants. Updates and "
            "deletes remain restricted to the owning project."
        ),
    },
}

_LLM_PROMPT_CONFIG_FIELDS: dict[str, Any] = {
    "validator_name": {
        "type": "string",
        "description": (
            "Which LLM-backed validator this prompt drives (e.g. "
            "`topic_relevance`, `answer_relevance_custom_llm`). Immutable "
            "after creation. Call GET /api/v1/guardrails for the current set."
        ),
    },
    "name": {"type": "string", "description": "Human-readable name."},
    "description": {"type": "string", "description": "What this prompt evaluates."},
    "prompt_schema_version": {
        "type": "integer",
        "default": 1,
        "description": (
            "Version of the prompt contract, for callers that iterate on "
            "wording. `topic_relevance_llm` reads it as: 1 = the prompt lists "
            "allowed topics, 2 = forbidden topics, 3 = both."
        ),
    },
    "llm_prompt": {
        "type": "string",
        "description": (
            "The prompt text. For `answer_relevance_custom_llm` this must "
            "contain both the `{query}` and `{answer}` placeholders; the "
            "guardrails service rejects it with a 422 otherwise."
        ),
    },
}

_VALIDATOR_CONFIG_FIELDS: dict[str, Any] = {
    "name": {
        "type": "string",
        "description": (
            "Unique name for this config within the project. Uniqueness is "
            "enforced on the name alone, so the same validator type may be "
            "registered more than once under different names."
        ),
    },
    "type": {
        "type": "string",
        "description": (
            "Which validator implementation to run. Call GET /api/v1/guardrails "
            "for the types this deployment supports and the tuning keys each "
            "one accepts."
        ),
    },
    "stage": {
        "type": "string",
        "enum": ["input", "output"],
        "description": (
            "Which text this config is intended for. Advisory only at run "
            "time: POST /guardrails routes on the text it is given, not on "
            "this field."
        ),
    },
    "on_fail_action": {
        "type": "string",
        "enum": ["exception", "fix", "rephrase"],
        "default": "fix",
        "description": (
            "Action taken when this validator fails. `fix` falls back to an "
            "empty string for validators with no programmatic repair."
        ),
    },
    "is_enabled": {
        "type": "boolean",
        "default": True,
        "description": "Whether this config is eligible to run.",
    },
}

_IDENTITY_FIELDS: dict[str, Any] = {
    "id": {"type": "string", "format": "uuid"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
}


def _optional(fields: dict[str, Any]) -> dict[str, Any]:
    """Same fields, none required — the PATCH shape. Omitted keys are
    forwarded as omitted, so they leave stored values untouched."""
    return {key: {**value, "nullable": True} for key, value in fields.items()}


_BAN_LIST = {
    "type": "object",
    "description": "A ban list as returned by the guardrails service.",
    "properties": {**_BAN_LIST_FIELDS, **_IDENTITY_FIELDS},
}

_LLM_PROMPT_CONFIG = {
    "type": "object",
    "description": "An LLM prompt config as returned by the guardrails service.",
    "properties": {
        **_LLM_PROMPT_CONFIG_FIELDS,
        "is_active": {"type": "boolean"},
        **_IDENTITY_FIELDS,
    },
}

_VALIDATOR_CONFIG = {
    "type": "object",
    "description": (
        "A validator config as returned by the guardrails service. The stored "
        "row is flattened together with its JSONB tuning config, so responses "
        "carry validator-specific keys beyond the ones listed here."
    ),
    "properties": {
        **_VALIDATOR_CONFIG_FIELDS,
        "organization_id": {"type": "integer"},
        "project_id": {"type": "integer"},
        **_IDENTITY_FIELDS,
    },
    "additionalProperties": True,
}

GUARDRAILS_COMPONENT_SCHEMAS: dict[str, dict[str, Any]] = {
    "GuardrailsBanListCreate": {
        "type": "object",
        "properties": _BAN_LIST_FIELDS,
        "required": ["name", "description", "banned_words", "domain"],
    },
    "GuardrailsBanListUpdate": {
        "type": "object",
        "description": "Omitted fields are left unchanged.",
        "properties": _optional(_BAN_LIST_FIELDS),
    },
    "GuardrailsBanListResponse": _envelope(_BAN_LIST),
    "GuardrailsBanListListResponse": _list_envelope(_BAN_LIST),
    "GuardrailsLLMPromptConfigCreate": {
        "type": "object",
        "properties": _LLM_PROMPT_CONFIG_FIELDS,
        "required": ["validator_name", "name", "description", "llm_prompt"],
    },
    "GuardrailsLLMPromptConfigUpdate": {
        "type": "object",
        "description": (
            "`validator_name` is absent by design — it is immutable after "
            "creation. `is_active` is settable only here; new configs are "
            "created active."
        ),
        "properties": {
            **_optional(
                {
                    key: value
                    for key, value in _LLM_PROMPT_CONFIG_FIELDS.items()
                    if key != "validator_name"
                }
            ),
            "is_active": {"type": "boolean", "nullable": True},
        },
    },
    "GuardrailsLLMPromptConfigResponse": _envelope(_LLM_PROMPT_CONFIG),
    "GuardrailsLLMPromptConfigListResponse": _list_envelope(_LLM_PROMPT_CONFIG),
    "GuardrailsValidatorConfigCreate": {
        "type": "object",
        "description": (
            "Extra keys are the validator's tuning config — e.g. "
            "`entity_types` for `pii_remover`, `slur_severity` for "
            "`content_safety`. Call GET /api/v1/guardrails for each type's "
            "accepted keys. `organization_id`/`project_id` must not be sent: "
            "the guardrails service derives the tenant from the authenticated "
            "context and rejects those keys with a 422."
        ),
        "properties": _VALIDATOR_CONFIG_FIELDS,
        "required": ["name", "type", "stage"],
        "additionalProperties": True,
    },
    "GuardrailsValidatorConfigUpdate": {
        "type": "object",
        "description": (
            "Only the base fields can be patched. The guardrails service "
            "forbids extra keys here, so validator-specific tuning cannot be "
            "changed through this route — delete and recreate the config "
            "instead. Extras are forwarded rather than dropped, so the "
            "upstream 422 is what the caller sees."
        ),
        "properties": _optional(_VALIDATOR_CONFIG_FIELDS),
        "additionalProperties": True,
    },
    "GuardrailsValidatorConfigResponse": _envelope(_VALIDATOR_CONFIG),
    "GuardrailsValidatorConfigListResponse": _list_envelope(_VALIDATOR_CONFIG),
    "GuardrailsValidatorTypeListResponse": {
        "type": "object",
        "description": (
            "The one guardrails route that is not wrapped in the standard "
            "`{success, data, ...}` envelope; the service returns this bare "
            "object."
        ),
        "properties": {
            "validators": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "Validator type discriminator.",
                        },
                        "config": {
                            "type": "object",
                            "description": (
                                "JSON Schema of the tuning fields this "
                                "validator accepts."
                            ),
                        },
                    },
                },
            }
        },
    },
    "GuardrailsDeleteResponse": _envelope(
        {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Human-readable confirmation.",
                }
            },
        }
    ),
    "GuardrailsErrorResponse": _envelope({"nullable": True}),
    # Failures raised by this gateway go through `app.utils.APIResponse`, which
    # carries an extra `errors` list the guardrails service's envelope has not.
    "GuardrailsGatewayErrorResponse": {
        **_envelope({"nullable": True}),
        "properties": {
            **_envelope({"nullable": True})["properties"],
            "errors": {
                "type": "array",
                "nullable": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
            },
        },
    },
}

# ---------- validator catalogue ----------
#
# One entry per validator type the guardrails service exposes, mirroring what
# `GET /api/v1/guardrails` returns at run time. Verified against
# `kaapi-guardrails` validator configs; that service stays the authority, so
# when the two disagree, believe the live response and fix this table.


class ConfigKey(NamedTuple):
    """One tuning key of a validator type.

    `schema` is what the live response carries for this key, trimmed of
    JSON Schema bookkeeping (`title`, `$defs`, `anyOf` wrappers around
    optionals). `description` is for humans and is not part of the response.
    """

    name: str
    schema: dict[str, Any]
    description: str


class ValidatorType(NamedTuple):
    type: str
    summary: str
    runs_on: str
    config: tuple[ConfigKey, ...] = ()
    notes: tuple[str, ...] = ()


class ValidatorGroup(NamedTuple):
    """A themed run of validator types, in the order they should be read."""

    title: str
    blurb: str
    types: tuple[str, ...]


# Rendering keeps only EXPOSED_VALIDATOR_TYPES.
GROUPS: tuple[ValidatorGroup, ...] = (
    ValidatorGroup(
        "Abuse, profanity and unsafe content",
        "Start with `content_safety`. The rest are narrower checks to add "
        "when it is not enough on its own.",
        (
            "content_safety",
            "uli_slur_match",
            "profanity_free",
            "nsfw_text",
            "llamaguard_7b",
            "gender_assumption_bias",
        ),
    ),
    ValidatorGroup(
        "Personal data and banned words",
        "These redact rather than block, so the conversation carries on with "
        "the sensitive words removed.",
        ("pii_remover", "ban_list"),
    ),
    ValidatorGroup(
        "Staying on topic, and answer quality",
        "These call an LLM, so each adds a little latency per message. "
        "Reach for `topic_relevance_llm` first.",
        (
            "topic_relevance_llm",
            "topic_relevance",
            "answer_relevance_custom_llm",
            "llm_critic",
        ),
    ),
)


_SEVERITY_LEVELS = ["low", "medium", "high", "all"]
_PII_ENTITY_TYPES = [
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "MEDICAL_LICENSE",
    "NRP",
    "PERSON",
    "PHONE_NUMBER",
    "URL",
    "IN_AADHAAR",
    "IN_PAN",
    "IN_PASSPORT",
    "IN_VEHICLE_REGISTRATION",
    "IN_VOTER",
]
_PII_DEFAULT_ENTITY_TYPES = [
    entity
    for entity in _PII_ENTITY_TYPES
    if entity not in ("LOCATION", "NRP", "PERSON")
]
_NEEDS_OPENAI_KEY = (
    "Needs an OpenAI key on the guardrails service. Without one the validator "
    "fails the request with an explicit error rather than passing silently."
)
_NO_PARTIAL_FIX = (
    "There is no partial repair: with `on_fail_action: fix` a failure returns "
    "an empty `safe_text`, and `metadata.reason` names this validator."
)

VALIDATOR_TYPES: tuple[ValidatorType, ...] = (
    ValidatorType(
        type="uli_slur_match",
        summary=(
            "Matches a curated Hindi/English slur list and redacts each hit "
            "with `[REDACTED_SLUR]`, leaving the rest of the message intact."
        ),
        runs_on="input and output",
        config=(
            ConfigKey(
                "languages",
                {"type": "array", "items": {"type": "string"}, "default": ["en", "hi"]},
                "Slur lists to load. `en` and `hi` are the curated ones.",
            ),
            ConfigKey(
                "severity",
                {"type": "string", "enum": _SEVERITY_LEVELS, "default": "all"},
                "Lowest severity to act on. `all` catches everything.",
            ),
        ),
    ),
    ValidatorType(
        type="profanity_free",
        summary="Flags profanity anywhere in the text.",
        runs_on="input and output",
        notes=(
            _NO_PARTIAL_FIX,
            "If you want slurs redacted *and* profanity checked, use "
            "`content_safety` instead of running both — it orders them so a "
            "redacted slur does not then trip the profanity check.",
        ),
    ),
    ValidatorType(
        type="content_safety",
        summary=(
            "Slur redaction followed by a profanity check, as one validator. "
            "The slur pass runs first, so text it has already cleaned is not "
            "blanked by the profanity pass."
        ),
        runs_on="input and output",
        config=(
            ConfigKey(
                "slur_severity",
                {"type": "string", "enum": _SEVERITY_LEVELS, "default": "all"},
                "Lowest slur severity to redact.",
            ),
            ConfigKey(
                "profanity_threshold",
                {"type": "number", "minimum": 0, "maximum": 1, "default": 0.9},
                (
                    "Profanity probability at or above which the text fails "
                    "and is blanked. Lower it to be stricter."
                ),
            ),
        ),
    ),
    ValidatorType(
        type="pii_remover",
        summary=("Replaces personal data with placeholders such as `<PHONE_NUMBER>`."),
        runs_on="input and output",
        config=(
            ConfigKey(
                "entity_types",
                {
                    "type": "array",
                    "items": {"type": "string", "enum": _PII_ENTITY_TYPES},
                    "default": _PII_DEFAULT_ENTITY_TYPES,
                },
                (
                    "Which entities to mask. Anything outside the listed "
                    "values is rejected when the config is saved."
                ),
            ),
        ),
        notes=(
            "`PERSON`, `LOCATION` and `NRP` are left out of the default on "
            "purpose: they come from an English-only name model that misfires "
            "on Indic names. Add them only if you have checked the result on "
            "your own traffic. Every other entity is pattern-matched.",
        ),
    ),
    ValidatorType(
        type="ban_list",
        summary="Redacts words you supply — product names, competitors, anything.",
        runs_on="input and output",
        config=(
            ConfigKey(
                "banned_words",
                {"type": "array", "items": {"type": "string"}, "default": None},
                "The words, inline.",
            ),
            ConfigKey(
                "ban_list_id",
                {"type": "string", "format": "uuid", "default": None},
                (
                    "A list stored through `/guardrails/ban_lists`, resolved "
                    "to its words at run time."
                ),
            ),
        ),
        notes=(
            "Give exactly one of the two. A config with neither is rejected. "
            "`ban_list_id` is the maintainable choice: edit the list once and "
            "every validator using it follows.",
        ),
    ),
    ValidatorType(
        type="gender_assumption_bias",
        summary=(
            "Flags text that assumes someone's gender — 'ask her doctor' when "
            "nothing said the patient is a woman."
        ),
        runs_on="input and output",
        config=(
            ConfigKey(
                "categories",
                {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["generic", "healthcare", "education", "all"],
                    },
                    "default": ["all"],
                },
                "Which domains to check. `all` enables every category.",
            ),
        ),
    ),
    ValidatorType(
        type="nsfw_text",
        summary="Classifies sexual/explicit content with a local model.",
        runs_on="input and output",
        config=(
            ConfigKey(
                "threshold",
                {"type": "number", "default": 0.8},
                "Probability above which text counts as NSFW.",
            ),
            ConfigKey(
                "validation_method",
                {"type": "string", "default": "sentence"},
                (
                    "`sentence` scores each sentence and fails if any one "
                    "crosses the threshold; `full` scores the message as a "
                    "whole, which is faster on short messages."
                ),
            ),
            ConfigKey(
                "device",
                {"type": "string", "default": "cpu"},
                "`cpu` or `cuda`.",
            ),
            ConfigKey(
                "model_name",
                {"type": "string", "default": "michellejieli/NSFW_text_classifier"},
                (
                    "HuggingFace classifier. It must emit an `NSFW` label — a "
                    "model that labels its classes differently never fails, "
                    "whatever it scores."
                ),
            ),
        ),
        notes=(
            _NO_PARTIAL_FIX,
            "Runs a transformer locally: budget roughly 200–500 ms per "
            "request on CPU. The default model is English-focused.",
        ),
    ),
    ValidatorType(
        type="llamaguard_7b",
        summary=(
            "Classifies text as safe or unsafe against LlamaGuard's policy "
            "set (violence and hate, sexual content, criminal planning, "
            "weapons, drugs, self-harm)."
        ),
        runs_on="input and output",
        config=(
            ConfigKey(
                "policies",
                {"type": "array", "items": {"type": "string"}, "default": None},
                (
                    "Subset to enforce, from `no_violence_hate`, "
                    "`no_sexual_content`, `no_criminal_planning`, "
                    "`no_guns_and_illegal_weapons`, `no_illegal_drugs`, "
                    "`no_encourage_self_harm`. Omit for all of them."
                ),
            ),
        ),
        notes=(
            "Runs remotely on the Guardrails Hub, so the guardrails service "
            "must have remote inferencing enabled.",
            _NO_PARTIAL_FIX,
            "Expect false positives on news, clinical and legal text, which "
            "discuss these topics without endorsing them.",
        ),
    ),
    ValidatorType(
        type="llm_critic",
        summary=(
            "Scores the text against metrics you write, using an LLM as the "
            "judge. The escape hatch for checks no other validator covers."
        ),
        runs_on="input or output",
        config=(
            ConfigKey(
                "metrics",
                {"type": "object"},
                (
                    "Metric name to a plain-English description of "
                    'what earns a high score, e.g. `{"clarity": "Is the '
                    'answer easy to follow?"}`.'
                ),
            ),
            ConfigKey(
                "max_score",
                {"type": "integer"},
                "Top of the scoring scale, e.g. `5`.",
            ),
            ConfigKey(
                "llm_callable",
                {"type": "string"},
                "Judge model, e.g. `gpt-4o-mini`.",
            ),
        ),
        notes=(
            "All three keys are required — unlike every other type, there is "
            "no default to fall back on.",
            _NEEDS_OPENAI_KEY,
            "LLM judges vary run to run. Keep `max_score` small and read a "
            "sample of verdicts before trusting it in production.",
        ),
    ),
    ValidatorType(
        type="topic_relevance_llm",
        summary=(
            "Keeps the conversation inside your subject with one LLM call, "
            "scoring the message 1–3 and failing below `threshold`."
        ),
        runs_on="input",
        config=(
            ConfigKey(
                "configuration",
                {"type": "string", "default": None},
                (
                    "The topic text itself. Read as *forbidden* topics by "
                    "default — see `prompt_schema_version`."
                ),
            ),
            ConfigKey(
                "prompt_schema_version",
                {"type": "integer", "minimum": 1, "default": 2},
                (
                    "How `configuration` is read: `1` = the topics you allow, "
                    "`2` = the topics you forbid, `3` = both (forbidden wins)."
                ),
            ),
            ConfigKey(
                "threshold",
                {"type": "integer", "minimum": 1, "maximum": 3, "default": 2},
                (
                    "Minimum score to pass. `2` lets ambiguous messages "
                    "through; `3` blocks them too."
                ),
            ),
            ConfigKey(
                "fallback_message",
                {"type": "string", "minLength": 1, "default": None},
                (
                    "What to say when the message is off-topic. The same LLM "
                    "call translates it into the user's language, so a Hindi "
                    "message gets a Hindi reply. Only used with "
                    "`on_fail_action: rephrase`."
                ),
            ),
            ConfigKey(
                "topic_relevance_config_id",
                {"type": "string", "format": "uuid", "default": None},
                (
                    "Use a prompt stored through `/guardrails/llm_prompt_configs` "
                    "instead of inline `configuration`."
                ),
            ),
            ConfigKey(
                "llm_callable",
                {"type": "string", "default": "gpt-4o-mini"},
                "Model that does the scoring.",
            ),
        ),
        notes=(
            "Give either `configuration` or `topic_relevance_config_id`.",
            "Pair it with `on_fail_action: rephrase`. With `fix` the user "
            "gets an empty string, since there is nothing to repair.",
            _NEEDS_OPENAI_KEY,
        ),
    ),
    ValidatorType(
        type="topic_relevance",
        summary=(
            "The older topic check, scored by `llm_critic` under the hood. "
            "Prefer `topic_relevance_llm` for new work: one call, a clearer "
            "score, and a localized fallback reply."
        ),
        runs_on="input",
        config=(
            ConfigKey(
                "topic_relevance_config_id",
                {"type": "string", "format": "uuid", "default": None},
                (
                    "Required. A prompt stored through "
                    "`/guardrails/llm_prompt_configs`; its text and schema "
                    "version are loaded at run time."
                ),
            ),
            ConfigKey(
                "configuration",
                {"type": "string", "default": None},
                "Filled in from the stored prompt; you do not send it.",
            ),
            ConfigKey(
                "prompt_schema_version",
                {"type": "integer", "default": None},
                "Also filled in from the stored prompt.",
            ),
            ConfigKey(
                "llm_callable",
                {"type": "string", "default": "gpt-4o-mini"},
                "Model that does the scoring.",
            ),
        ),
        notes=(
            "The stored prompt must contain the `{{TOPIC_CONFIGURATION}}` "
            "placeholder.",
            _NEEDS_OPENAI_KEY,
        ),
    ),
    ValidatorType(
        type="answer_relevance_custom_llm",
        summary=(
            "Judges whether an answer actually addresses the question that "
            "was asked."
        ),
        runs_on="output",
        config=(
            ConfigKey(
                "custom_prompt_id",
                {"type": "string", "format": "uuid", "default": None},
                (
                    "A scoring prompt stored through "
                    "`/guardrails/llm_prompt_configs`."
                ),
            ),
            ConfigKey(
                "prompt_template",
                {"type": "string", "default": None},
                (
                    "The same thing inline. Must contain the `{query}` and "
                    "`{answer}` placeholders."
                ),
            ),
            ConfigKey(
                "llm_callable",
                {"type": "string", "default": "gpt-4o-mini"},
                "Model that does the judging.",
            ),
            ConfigKey(
                "input",
                {"type": "string", "default": ""},
                (
                    "Filled in by the service from the guardrails request. "
                    "Listed because it appears in the schema; do not send it."
                ),
            ),
            ConfigKey(
                "output",
                {"type": "string", "default": ""},
                "Filled in by the service too. Do not send it.",
            ),
        ),
        notes=(
            "Give either `custom_prompt_id` or `prompt_template`.",
            "This validator reads the question and the answer together, so a "
            "guardrails run using it must send `input` as a JSON string: "
            '`{"query": "...", "answer": "..."}`.',
            _NEEDS_OPENAI_KEY,
        ),
    ),
)


def _type_label(schema: dict[str, Any]) -> str:
    if "enum" in schema:
        return "one of " + ", ".join(f"`{value}`" for value in schema["enum"])
    if schema.get("type") == "array":
        items = schema.get("items", {})
        if "enum" in items:
            return "array of " + ", ".join(f"`{value}`" for value in items["enum"])
        return f"array of {items.get('type', 'any')}"
    if schema.get("format") == "uuid":
        return "uuid"
    label = schema.get("type", "any")
    bounds = [
        f"{name} {schema[key]}"
        for key, name in (
            ("minimum", "min"),
            ("maximum", "max"),
            ("minLength", "min length"),
        )
        if key in schema
    ]
    return f"{label} ({', '.join(bounds)})" if bounds else label


def _default_label(schema: dict[str, Any]) -> str:
    if "default" not in schema:
        return "**required**"
    default = schema["default"]
    if default is None:
        return "unset"
    if isinstance(default, list) and len(default) > 4:
        return f"all {len(default)} pattern-based types"
    return f"`{json.dumps(default)}`"


def _by_type() -> dict[str, ValidatorType]:
    catalogued = {validator.type: validator for validator in VALIDATOR_TYPES}
    grouped = {name for group in GROUPS for name in group.types}
    # A type missing from GROUPS would silently vanish from the docs, which is
    # worse than a loud failure at import.
    if catalogued.keys() != grouped:
        raise RuntimeError(
            "VALIDATOR_TYPES and GROUPS disagree: "
            f"{sorted(catalogued.keys() ^ grouped)}"
        )
    return catalogued


def _ordered_types() -> list[ValidatorType]:
    """Exposed types in `GROUPS` order."""
    catalogued = _by_type()
    return [
        catalogued[name]
        for group in GROUPS
        for name in group.types
        if name in EXPOSED_VALIDATOR_TYPES
    ]


def _render_validator(validator: ValidatorType) -> list[str]:
    lines = [
        f"#### `{validator.type}`",
        "",
        f"{validator.summary} Runs on **{validator.runs_on}**.",
        "",
    ]
    if validator.config:
        lines += ["| Key | Type | Default | What it does |", "|---|---|---|---|"]
        lines += [
            f"| `{key.name}` | {_type_label(key.schema)} | "
            f"{_default_label(key.schema)} | {key.description} |"
            for key in validator.config
        ]
    else:
        lines.append(
            "No tuning keys of its own — the shared fields are the whole config."
        )
    if validator.notes:
        lines.append("")
        lines += [f"- {note}" for note in validator.notes]
    return lines


def render_validator_reference() -> str:
    """Render the catalogue as Markdown, appended to the route description.

    Generated rather than hand-written so the reference and the example
    response in this module cannot drift apart.
    """
    lines = [
        "",
        "<br>",
        "",
        "## The validator types",
        "",
        "Grouped by the problem they solve. Every one also takes the shared "
        "fields above.",
    ]
    catalogued = _by_type()
    for group in GROUPS:
        exposed = [name for name in group.types if name in EXPOSED_VALIDATOR_TYPES]
        if not exposed:
            continue
        lines += ["", "<br>", "", f"### {group.title}", "", group.blurb]
        for name in exposed:
            lines += ["", "---", ""]
            lines += _render_validator(catalogued[name])
    return "\n".join(lines) + "\n"


def _validator_catalogue_example() -> dict[str, Any]:
    """The catalogue in the shape `GET /api/v1/guardrails` returns it."""
    return {
        "validators": [
            {
                "type": validator.type,
                "config": {
                    "type": "object",
                    "properties": {key.name: key.schema for key in validator.config},
                },
            }
            for validator in _ordered_types()
        ]
    }


# ---------- examples ----------

BAN_LIST_CREATE_EXAMPLE = {
    "name": "Safety Banned Terms",
    "description": "Terms not allowed under this tenant's policy",
    "banned_words": ["scam", "fraud"],
    "domain": "abuse",
    "is_public": False,
}
BAN_LIST_UPDATE_EXAMPLE = {"banned_words": ["scam", "fraud", "phishing"]}
_BAN_LIST_EXAMPLE = {
    **BAN_LIST_CREATE_EXAMPLE,
    "id": _UUID_EXAMPLE,
    "created_at": _TIMESTAMP_EXAMPLE,
    "updated_at": _TIMESTAMP_EXAMPLE,
}

LLM_PROMPT_CONFIG_CREATE_EXAMPLES: dict[str, tuple[str, Any]] = {
    "topic_relevance": (
        "Topic scope prompt",
        {
            "validator_name": "topic_relevance",
            "name": "Maternal Health Scope",
            "description": "Topic guard for a maternal health support bot",
            "prompt_schema_version": 2,
            "llm_prompt": "Forbidden topics: politics, religion, sports.",
        },
    ),
    "answer_relevance_custom_llm": (
        "Scoring prompt — must contain {query} and {answer}",
        {
            "validator_name": "answer_relevance_custom_llm",
            "name": "Support Answer Grader",
            "description": "Grades how well the answer addresses the query",
            "prompt_schema_version": 1,
            "llm_prompt": (
                "Query: {query}\nAnswer: {answer}\n" "Score the relevance from 1 to 5."
            ),
        },
    ),
}
LLM_PROMPT_CONFIG_UPDATE_EXAMPLE = {
    "description": "Updated scope",
    "is_active": True,
}
_LLM_PROMPT_CONFIG_EXAMPLE = {
    "id": _UUID_EXAMPLE,
    **LLM_PROMPT_CONFIG_CREATE_EXAMPLES["topic_relevance"][1],
    "is_active": True,
    "created_at": _TIMESTAMP_EXAMPLE,
    "updated_at": _TIMESTAMP_EXAMPLE,
}

VALIDATOR_CONFIG_CREATE_EXAMPLES: dict[str, tuple[str, Any]] = {
    "pii_remover": (
        "PII masking — entity_types is validator-specific tuning",
        {
            "name": "PII Redaction Input",
            "type": "pii_remover",
            "stage": "input",
            "on_fail_action": "fix",
            "is_enabled": True,
            "entity_types": ["EMAIL_ADDRESS", "PHONE_NUMBER", "IN_AADHAAR"],
        },
    ),
    "content_safety": (
        "Slur redaction plus profanity check",
        {
            "name": "Content Safety Input",
            "type": "content_safety",
            "stage": "input",
            "on_fail_action": "fix",
            "slur_severity": "all",
            "profanity_threshold": 0.9,
        },
    ),
    "ban_list": (
        "Ban list by id, resolved upstream",
        {
            "name": "Abuse Terms Input",
            "type": "ban_list",
            "stage": "input",
            "on_fail_action": "fix",
            "ban_list_id": _UUID_EXAMPLE,
        },
    ),
    "topic_relevance_llm": (
        "Topic scope with a localized fallback reply",
        {
            "name": "Farming Scope Input",
            "type": "topic_relevance_llm",
            "stage": "input",
            "on_fail_action": "rephrase",
            "configuration": "politics, religion, sports",
            "prompt_schema_version": 2,
            "fallback_message": "I can only help with farming questions.",
        },
    ),
}
VALIDATOR_CONFIG_UPDATE_EXAMPLE = {"is_enabled": False, "on_fail_action": "rephrase"}
_VALIDATOR_CONFIG_EXAMPLE = {
    "id": _UUID_EXAMPLE,
    "organization_id": 1,
    "project_id": 1,
    **VALIDATOR_CONFIG_CREATE_EXAMPLES["pii_remover"][1],
    "created_at": _TIMESTAMP_EXAMPLE,
    "updated_at": _TIMESTAMP_EXAMPLE,
}


def _success(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data, "error": None, "metadata": None}


def _failure(message: str) -> dict[str, Any]:
    """Failure body as the guardrails service writes it, echoed verbatim."""
    return {"success": False, "data": None, "error": message, "metadata": None}


def _gateway_failure(message: str) -> dict[str, Any]:
    """Failure body as this gateway writes it; `errors` is only populated for
    request-validation failures."""
    return {**_failure(message), "errors": None}


# ---------- documented responses ----------

# Gateway-side failures: the request never reaches the guardrails service.
UNAUTHORIZED = ResponseDoc(
    "Missing or invalid API key.",
    _gateway_failure("Not authenticated"),
    ref("GuardrailsGatewayErrorResponse"),
)
FORBIDDEN = ResponseDoc(
    "The API key's project is not permitted to use guardrails.",
    _gateway_failure("Permission denied"),
    ref("GuardrailsGatewayErrorResponse"),
)
BAD_GATEWAY = ResponseDoc(
    "The guardrails service was unreachable or returned a non-JSON body.",
    _gateway_failure("Guardrails service unavailable"),
    ref("GuardrailsGatewayErrorResponse"),
)
JOB_NOT_FOUND = ResponseDoc(
    "No guardrails job with this id in your project. Also returned for a job "
    "of another type, so job ids cannot be probed across endpoints.",
    _gateway_failure("Job not found"),
    ref("GuardrailsGatewayErrorResponse"),
)

# The async job routes run locally, so only the gateway's own auth failures
# apply; their 200 schemas come from the Pydantic models they still validate.
JOB_RESPONSES = {401: UNAUTHORIZED, 403: FORBIDDEN}
JOB_ITEM_RESPONSES = {**JOB_RESPONSES, 404: JOB_NOT_FOUND}
# Echoed verbatim from the guardrails service, which owns validation.
UPSTREAM_NOT_FOUND = ResponseDoc(
    "No such record in this project (echoed from the guardrails service).",
    _failure("Not found"),
    ref("GuardrailsErrorResponse"),
)
UPSTREAM_VALIDATION_ERROR = ResponseDoc(
    "Rejected by the guardrails service, which owns validation for these "
    "payloads. The gateway forwards bodies unchanged.",
    _failure("Missing required field(s): name"),
    ref("GuardrailsErrorResponse"),
)

PROXY_READ_RESPONSES = {
    401: UNAUTHORIZED,
    403: FORBIDDEN,
    502: BAD_GATEWAY,
}
PROXY_ITEM_RESPONSES = {**PROXY_READ_RESPONSES, 404: UPSTREAM_NOT_FOUND}
PROXY_WRITE_RESPONSES = {
    **PROXY_READ_RESPONSES,
    422: UPSTREAM_VALIDATION_ERROR,
}
PROXY_ITEM_WRITE_RESPONSES = {**PROXY_ITEM_RESPONSES, 422: UPSTREAM_VALIDATION_ERROR}


def _ok(description: str, example: Any, schema_name: str) -> ResponseDoc:
    return ResponseDoc(description, example, ref(schema_name))


BAN_LIST_OK = _ok(
    "The ban list.", _success(_BAN_LIST_EXAMPLE), "GuardrailsBanListResponse"
)
BAN_LIST_LIST_OK = _ok(
    "Ban lists visible to this project.",
    _success([_BAN_LIST_EXAMPLE]),
    "GuardrailsBanListListResponse",
)
LLM_PROMPT_CONFIG_OK = _ok(
    "The LLM prompt config.",
    _success(_LLM_PROMPT_CONFIG_EXAMPLE),
    "GuardrailsLLMPromptConfigResponse",
)
LLM_PROMPT_CONFIG_LIST_OK = _ok(
    "LLM prompt configs for this project.",
    _success([_LLM_PROMPT_CONFIG_EXAMPLE]),
    "GuardrailsLLMPromptConfigListResponse",
)
VALIDATOR_CONFIG_OK = _ok(
    "The validator config, flattened with its tuning keys.",
    _success(_VALIDATOR_CONFIG_EXAMPLE),
    "GuardrailsValidatorConfigResponse",
)
VALIDATOR_CONFIG_LIST_OK = _ok(
    "Validator configs for this project.",
    _success([_VALIDATOR_CONFIG_EXAMPLE]),
    "GuardrailsValidatorConfigListResponse",
)
VALIDATOR_TYPE_LIST_OK = _ok(
    "Every validator type this deployment supports, with the JSON Schema of "
    "its tuning keys. Trimmed here for readability: the live response also "
    "carries JSON Schema bookkeeping (`title`, `$defs`, `anyOf` wrappers "
    "around optional fields) and the shared `type`, `stage` and `on_fail` "
    "fields on every entry.",
    _validator_catalogue_example(),
    "GuardrailsValidatorTypeListResponse",
)
DELETE_OK = _ok(
    "Deleted. The guardrails service answers deletes with a confirmation "
    "body rather than a 204.",
    _success({"message": "Ban list deleted successfully"}),
    "GuardrailsDeleteResponse",
)
