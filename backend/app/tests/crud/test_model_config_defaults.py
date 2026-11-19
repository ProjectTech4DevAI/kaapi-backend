"""Every production default must have an active, priced model_config row.

Request-time validation only warns, so a default with no row fails silently: no effort
handling and no cost estimate.
"""

import pytest
from sqlmodel import Session

from app.core.config import settings
from app.crud.evaluations.embeddings import EMBEDDING_MODEL
from app.crud.model_config import get_model_config
from app.models.config.assessment_blob import (
    DEFAULT_PREFILTER_MODEL,
    DEFAULT_PREFILTER_PROVIDER,
)
from app.models.llm.constants import DEFAULT_RAG_MODEL, DEFAULT_TEXT_MODELS

# (provider, model_name, where the default lives). Google text defaults are seeded by 063.
PRODUCTION_DEFAULTS: list[tuple[str, str, str]] = [
    ("openai", DEFAULT_TEXT_MODELS["openai"], "DEFAULT_TEXT_MODELS['openai']"),
    ("anthropic", DEFAULT_TEXT_MODELS["anthropic"], "DEFAULT_TEXT_MODELS['anthropic']"),
    ("openai", DEFAULT_RAG_MODEL, "DEFAULT_RAG_MODEL"),
    ("openai", settings.EVAL_JUDGE_MODEL, "settings.EVAL_JUDGE_MODEL"),
    ("anthropic", settings.EVAL_SUMMARY_MODEL, "settings.EVAL_SUMMARY_MODEL"),
    (
        "anthropic",
        settings.PROMPT_IMPROVEMENT_MODEL,
        "settings.PROMPT_IMPROVEMENT_MODEL",
    ),
    (
        DEFAULT_PREFILTER_PROVIDER.value,
        DEFAULT_PREFILTER_MODEL,
        "DEFAULT_PREFILTER_MODEL",
    ),
    ("openai", EMBEDDING_MODEL, "EMBEDDING_MODEL"),
]


@pytest.mark.parametrize(
    ("provider", "model_name", "source"),
    PRODUCTION_DEFAULTS,
    ids=[source for _, _, source in PRODUCTION_DEFAULTS],
)
def test_production_default_has_active_priced_row(
    db: Session, provider: str, model_name: str, source: str
) -> None:
    # include_inactive so a deactivated row reports as deactivated, not missing.
    row = get_model_config(
        session=db, provider=provider, model_name=model_name, include_inactive=True
    )

    assert row is not None, (
        f"{source} is '{model_name}' but global.model_config has no "
        f"(provider='{provider}', model_name='{model_name}') row — add one in a migration."
    )
    assert row.is_active, f"{source} is '{model_name}', whose row is is_active=false."
    assert row.pricing, f"{source} is '{model_name}', whose row has no pricing."


def test_eval_judge_effort_is_honoured_by_its_model(db: Session) -> None:
    row = get_model_config(
        session=db, provider="openai", model_name=settings.EVAL_JUDGE_MODEL
    )
    assert row is not None

    effort = row.config.get("effort")
    assert effort is not None, (
        f"EVAL_JUDGE_MODEL '{settings.EVAL_JUDGE_MODEL}' has no 'effort' key in its config, "
        "so is_reasoning_model() is False and the mapper suppresses the judge's effort."
    )

    options = effort.get("options", [])
    assert settings.EVAL_JUDGE_REASONING_EFFORT in options, (
        f"EVAL_JUDGE_REASONING_EFFORT='{settings.EVAL_JUDGE_REASONING_EFFORT}' is not one of "
        f"the levels '{settings.EVAL_JUDGE_MODEL}' accepts ({options})."
    )
