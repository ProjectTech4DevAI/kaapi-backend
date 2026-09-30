"""Enums mirroring the kaapi-guardrails service contract.

These exist so Swagger renders real dropdowns instead of free-form strings on
the proxy routes. They are a *mirror*, not the source of truth: the upstream
service owns validation, and any value it accepts but we reject would turn a
working call into a spurious 422. Keep these in sync with
`kaapi-guardrails/backend/app/core/enum.py`.
"""

from enum import Enum


class ValidatorTypeEnum(str, Enum):
    """Validator implementations the guardrails service can run."""

    LEXICAL_SLUR = "uli_slur_match"
    PII_REMOVER = "pii_remover"
    GENDER_ASSUMPTION_BIAS = "gender_assumption_bias"
    BAN_LIST = "ban_list"
    TOPIC_RELEVANCE = "topic_relevance"
    TOPIC_RELEVANCE_LLM = "topic_relevance_llm"
    LLM_CRITIC = "llm_critic"
    LLAMAGUARD_7B = "llamaguard_7b"
    PROFANITY_FREE = "profanity_free"
    NSFW_TEXT = "nsfw_text"
    ANSWER_RELEVANCE_CUSTOM_LLM = "answer_relevance_custom_llm"


class StageEnum(str, Enum):
    """Which side of an LLM exchange a validator is meant to inspect."""

    INPUT = "input"
    OUTPUT = "output"


class GuardrailOnFailEnum(str, Enum):
    """What the service does when a validator fails.

    ``fix`` falls back to an empty string for validators that have no
    programmatic repair (e.g. ``profanity_free``).
    """

    EXCEPTION = "exception"
    FIX = "fix"
    REPHRASE = "rephrase"


class LLMValidatorNameEnum(str, Enum):
    """Validators that are driven by a stored LLM prompt config."""

    TOPIC_RELEVANCE = "topic_relevance"
    ANSWER_RELEVANCE_CUSTOM_LLM = "answer_relevance_custom_llm"
