"""
Cost tracking for evaluation runs.

Token usage is aggregated per stage (response generation, embedding) and
priced against `global.model_config` using OpenAI Batch rates. Failures
here must never block evaluation completion — `attach_cost` swallows
exceptions and logs a warning.

Persisted shape on `eval_run.cost`:

    {
        "response":          {model, input_tokens, output_tokens, total_tokens, cost_usd},
        "embedding":         {model, input_tokens, output_tokens, total_tokens, cost_usd},
        "judge":             {model, input_tokens, output_tokens, total_tokens, cost_usd},
        "total_cost_usd": float,
    }

Any stage entry is optional. Embedding entries use output_tokens=0. One combined call
grades every metric, so judge tokens can't be split per metric and form a single stage,
priced from per-row usage like the response stage.
"""

import logging
from collections.abc import Callable, Iterable
from typing import Any

from sqlmodel import Session

from app.crud.model_config import estimate_model_cost
from app.models import EvaluationRun

logger = logging.getLogger(__name__)

# USD rounding precision for persisted cost values.
COST_USD_DECIMALS = 6


def _cost_usd(estimate: dict[str, Any] | None) -> float:
    """Sum the per-direction costs from an estimate and round to our USD precision."""
    if not estimate:
        return 0.0
    total = float(estimate.get("input_cost", 0.0)) + float(
        estimate.get("output_cost", 0.0)
    )
    return round(total, COST_USD_DECIMALS)


def _sum_tokens(
    items: Iterable[dict[str, Any]],
    usage_extractor: Callable[[dict[str, Any]], dict[str, Any] | None],
    input_key: str,
) -> dict[str, int]:
    """Sum (input, output, total) tokens across items using a per-item usage extractor.

    The OpenAI Embeddings API reports input tokens as ``prompt_tokens`` and has
    no output tokens; chat/responses APIs use ``input_tokens`` and ``output_tokens``.
    Missing keys default to 0, so the embedding case naturally produces
    output_tokens=0.
    """
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for item in items:
        usage = usage_extractor(item)
        if not usage:
            continue
        totals["input_tokens"] += usage.get(input_key, 0)
        totals["output_tokens"] += usage.get("output_tokens", 0)
        totals["total_tokens"] += usage.get("total_tokens", 0)
    return totals


def _build_cost_entry(
    session: Session,
    model: str,
    totals: dict[str, int],
) -> dict[str, Any]:
    """Price aggregated token usage against the model's batch pricing row."""
    estimate = estimate_model_cost(
        session=session,
        provider="openai",
        model_name=model,
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        usage_type="batch",
    )
    return {
        "model": model,
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "total_tokens": totals["total_tokens"],
        "cost_usd": _cost_usd(estimate),
    }


def _build_response_cost_entry(
    session: Session, model: str, results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a response-stage cost entry from parsed evaluation results."""
    totals = _sum_tokens(
        items=results,
        usage_extractor=lambda r: r.get("usage"),
        input_key="input_tokens",
    )
    return _build_cost_entry(session=session, model=model, totals=totals)


def _build_embedding_cost_entry(
    session: Session, model: str, raw_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build an embedding-stage cost entry from raw embedding batch output."""
    totals = _sum_tokens(
        items=raw_results,
        usage_extractor=lambda r: r.get("response", {}).get("body", {}).get("usage"),
        input_key="prompt_tokens",
    )
    return _build_cost_entry(session=session, model=model, totals=totals)


def _build_judge_cost_entry(
    session: Session, model: str, results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a judge-stage cost entry from per-row judge usage results."""
    totals = _sum_tokens(
        items=results,
        usage_extractor=lambda r: r.get("usage"),
        input_key="input_tokens",
    )
    return _build_cost_entry(session=session, model=model, totals=totals)


def _build_cost_dict(
    response_entry: dict[str, Any] | None,
    embedding_entry: dict[str, Any] | None,
    judge_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine per-stage entries into the `eval_run.cost` payload with a grand total."""
    cost: dict[str, Any] = {}
    total = 0.0

    if response_entry:
        cost["response"] = response_entry
        total += response_entry.get("cost_usd", 0.0)

    if embedding_entry:
        cost["embedding"] = embedding_entry
        total += embedding_entry.get("cost_usd", 0.0)

    if judge_entry:
        cost["judge"] = judge_entry
        total += judge_entry.get("cost_usd", 0.0)

    cost["total_cost_usd"] = round(total, COST_USD_DECIMALS)
    return cost


def attach_cost(
    session: Session,
    eval_run: EvaluationRun,
    log_prefix: str,
    *,
    response_model: str | None = None,
    response_results: list[dict[str, Any]] | None = None,
    embedding_model: str | None = None,
    embedding_raw_results: list[dict[str, Any]] | None = None,
    judge_model: str | None = None,
    judge_results: list[dict[str, Any]] | None = None,
) -> None:
    """Compute cost for the given stage(s) and attach to `eval_run.cost`, never raising.

    Caller is responsible for persisting `eval_run` afterwards. Any stage's
    previously-computed entry on `eval_run.cost` is preserved when that stage's
    inputs are not supplied, so partial updates never clobber prior data.
    """
    try:
        existing_cost = eval_run.cost or {}

        if response_model is not None and response_results is not None:
            response_entry = _build_response_cost_entry(
                session=session, model=response_model, results=response_results
            )
        else:
            response_entry = existing_cost.get("response")

        if embedding_model is not None and embedding_raw_results is not None:
            embedding_entry = _build_embedding_cost_entry(
                session=session,
                model=embedding_model,
                raw_results=embedding_raw_results,
            )
        else:
            embedding_entry = existing_cost.get("embedding")

        if judge_model is not None and judge_results is not None:
            judge_entry = _build_judge_cost_entry(
                session=session, model=judge_model, results=judge_results
            )
        else:
            judge_entry = existing_cost.get("judge")

        eval_run.cost = _build_cost_dict(
            response_entry=response_entry,
            embedding_entry=embedding_entry,
            judge_entry=judge_entry,
        )
    except Exception as cost_err:
        logger.warning(
            f"[attach_cost] {log_prefix} Failed to compute cost | {cost_err}"
        )
