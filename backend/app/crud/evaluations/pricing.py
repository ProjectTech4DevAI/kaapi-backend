"""
Pricing utilities for evaluation cost tracking.

This module provides model pricing data and cost calculation functions
for both response generation and embedding stages of evaluation runs.

Pricing uses OpenAI Batch API rates (50% cheaper than real-time).
Source: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

import logging
from collections.abc import Callable, Iterable
from typing import Any

from app.crud.evaluations.embeddings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Number of decimals to round USD cost values to.
COST_USD_DECIMALS = 6

# Batch API pricing in USD per token.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # GPT-4o (batch pricing)
    "gpt-4o": {
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 5e-06,
    },
    "gpt-4o-mini": {
        "input_cost_per_token": 7.5e-08,
        "output_cost_per_token": 3e-07,
    },
    # GPT-4.1 (batch pricing)
    "gpt-4.1": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 4e-06,
    },
    # GPT-5 (batch pricing)
    "gpt-5": {
        "input_cost_per_token": 6.25e-07,
        "output_cost_per_token": 5e-06,
    },
    "gpt-5-mini": {
        "input_cost_per_token": 1.25e-07,
        "output_cost_per_token": 1e-06,
    },
    "gpt-5-nano": {
        "input_cost_per_token": 2.5e-08,
        "output_cost_per_token": 2e-07,
    },
    # GPT-5.4 (batch pricing)
    "gpt-5.4": {
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 7.5e-06,
    },
    "gpt-5.4-pro": {
        "input_cost_per_token": 1.5e-05,
        "output_cost_per_token": 9e-05,
    },
    "gpt-5.4-mini": {
        "input_cost_per_token": 3.75e-07,
        "output_cost_per_token": 2.25e-06,
    },
    "gpt-5.4-nano": {
        "input_cost_per_token": 1e-07,
        "output_cost_per_token": 6.25e-07,
    },
    # Embedding models (batch pricing)
    EMBEDDING_MODEL: {
        "input_cost_per_token": 6.5e-08,
    },
}


def calculate_token_cost(
    model: str, input_tokens: int, output_tokens: int = 0
) -> float:
    """
    Calculate USD cost for a model call given input and output token counts.

    Used for both response generation (input + output tokens) and embeddings
    (input tokens only — pass output_tokens=0 or omit).

    Args:
        model: OpenAI model name (e.g., "gpt-4o", "text-embedding-3-large")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output tokens (default 0 for embeddings)

    Returns:
        Cost in USD. Returns 0.0 if model is unknown.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        logger.warning(
            f"[calculate_token_cost] Unknown model '{model}', returning cost 0.0"
        )
        return 0.0

    input_cost = input_tokens * pricing.get("input_cost_per_token", 0)
    output_cost = output_tokens * pricing.get("output_cost_per_token", 0)
    return input_cost + output_cost


def _sum_usage(
    items: Iterable[dict[str, Any]],
    usage_extractor: Callable[[dict[str, Any]], dict[str, Any] | None],
    fields: tuple[str, ...],
) -> dict[str, int]:
    """
    Sum named token fields across items, using a caller-supplied extractor
    to locate the per-item usage dict.

    Args:
        items: Iterable of items to aggregate
        usage_extractor: Function returning the usage dict for an item, or None
        fields: Token field names to sum (e.g., "input_tokens", "total_tokens")

    Returns:
        Mapping of field name to summed value
    """
    totals: dict[str, int] = {field: 0 for field in fields}
    for item in items:
        usage = usage_extractor(item)
        if not usage:
            continue
        for field in fields:
            totals[field] += usage.get(field, 0)
    return totals


def build_response_cost_entry(
    model: str, results: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Aggregate token usage from parsed evaluation results and calculate cost.

    Args:
        model: OpenAI model name used for response generation
        results: Parsed evaluation results from parse_evaluation_output(),
                 each containing a "usage" dict with input_tokens/output_tokens/total_tokens

    Returns:
        Response cost entry for the cost JSONB field
    """
    totals = _sum_usage(
        items=results,
        usage_extractor=lambda r: r.get("usage"),
        fields=("input_tokens", "output_tokens", "total_tokens"),
    )

    cost_usd = calculate_token_cost(
        model=model,
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
    )

    return {
        "model": model,
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "total_tokens": totals["total_tokens"],
        "cost_usd": round(cost_usd, COST_USD_DECIMALS),
    }


def build_embedding_cost_entry(
    model: str, raw_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Aggregate token usage from raw embedding batch results and calculate cost.

    Args:
        model: OpenAI embedding model name
        raw_results: Raw JSONL lines from embedding batch output,
                     each containing response.body.usage with prompt_tokens/total_tokens

    Returns:
        Embedding cost entry for the cost JSONB field
    """
    totals = _sum_usage(
        items=raw_results,
        usage_extractor=lambda r: r.get("response", {}).get("body", {}).get("usage"),
        fields=("prompt_tokens", "total_tokens"),
    )

    cost_usd = calculate_token_cost(model=model, input_tokens=totals["prompt_tokens"])

    return {
        "model": model,
        "prompt_tokens": totals["prompt_tokens"],
        "total_tokens": totals["total_tokens"],
        "cost_usd": round(cost_usd, COST_USD_DECIMALS),
    }


def build_cost_dict(
    response_entry: dict[str, Any] | None = None,
    embedding_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Combine response and embedding cost entries into the final cost JSONB structure.

    Args:
        response_entry: Response cost entry from build_response_cost_entry()
        embedding_entry: Embedding cost entry from build_embedding_cost_entry()

    Returns:
        Combined cost dict with total_cost_usd
    """
    cost: dict[str, Any] = {}

    response_cost = 0.0
    embedding_cost = 0.0

    if response_entry:
        cost["response"] = response_entry
        response_cost = response_entry.get("cost_usd", 0.0)

    if embedding_entry:
        cost["embedding"] = embedding_entry
        embedding_cost = embedding_entry.get("cost_usd", 0.0)

    cost["total_cost_usd"] = round(response_cost + embedding_cost, COST_USD_DECIMALS)

    return cost
