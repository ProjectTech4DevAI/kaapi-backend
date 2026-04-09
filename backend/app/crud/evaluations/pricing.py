"""
Pricing utilities for evaluation cost tracking.

This module provides model pricing data and cost calculation functions
for both response generation and embedding stages of evaluation runs.

Pricing uses OpenAI Batch API rates (50% cheaper than real-time).
Source: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Batch API pricing in USD per token
MODEL_PRICING: dict[str, dict[str, Any]] = {
    # Chat models (batch pricing)
    "gpt-4o": {
        "mode": "chat",
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 5e-06,
    },
    "gpt-4o-2024-08-06": {
        "mode": "chat",
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 5e-06,
    },
    "gpt-4o-mini": {
        "mode": "chat",
        "input_cost_per_token": 7.5e-08,
        "output_cost_per_token": 3e-07,
    },
    "gpt-4o-mini-2024-07-18": {
        "mode": "chat",
        "input_cost_per_token": 7.5e-08,
        "output_cost_per_token": 3e-07,
    },
    # Embedding models (batch pricing)
    "text-embedding-3-large": {
        "mode": "embedding",
        "input_cost_per_token": 6.5e-08,
    },
    "text-embedding-3-small": {
        "mode": "embedding",
        "input_cost_per_token": 1e-08,
    },
    "text-embedding-ada-002": {
        "mode": "embedding",
        "input_cost_per_token": 1e-07,
    },
}


def calculate_response_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate USD cost for response generation.

    Args:
        model: OpenAI model name (e.g., "gpt-4o")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD. Returns 0.0 if model is unknown.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        logger.warning(
            f"[calculate_response_cost] Unknown model '{model}', returning cost 0.0"
        )
        return 0.0

    input_cost = input_tokens * pricing.get("input_cost_per_token", 0)
    output_cost = output_tokens * pricing.get("output_cost_per_token", 0)
    return input_cost + output_cost


def calculate_embedding_cost(model: str, prompt_tokens: int) -> float:
    """
    Calculate USD cost for embeddings.

    Args:
        model: OpenAI embedding model name (e.g., "text-embedding-3-large")
        prompt_tokens: Number of prompt tokens

    Returns:
        Cost in USD. Returns 0.0 if model is unknown.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        logger.warning(
            f"[calculate_embedding_cost] Unknown model '{model}', returning cost 0.0"
        )
        return 0.0

    return prompt_tokens * pricing.get("input_cost_per_token", 0)


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
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for result in results:
        usage = result.get("usage")
        if not usage:
            continue
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
        total_tokens += usage.get("total_tokens", 0)

    cost_usd = calculate_response_cost(model, total_input_tokens, total_output_tokens)

    return {
        "model": model,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost_usd, 6),
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
    total_prompt_tokens = 0
    total_tokens = 0

    for response in raw_results:
        usage = response.get("response", {}).get("body", {}).get("usage")
        if not usage:
            continue
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_tokens += usage.get("total_tokens", 0)

    cost_usd = calculate_embedding_cost(model, total_prompt_tokens)

    return {
        "model": model,
        "prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost_usd, 6),
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

    cost["total_cost_usd"] = round(response_cost + embedding_cost, 6)

    return cost
