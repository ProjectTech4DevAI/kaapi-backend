"""Per-item result shapes for the fast evaluation stages.

Pure builders and aggregation helpers: no OpenAI, DB or storage access. The
dicts produced here are the units uploaded to S3, so they must stay
JSON-serializable and match the batch path's shape.
"""

from typing import Any

from app.core.config import settings
from app.crud.evaluations.response_parsing import field_value

RESPONSE_USAGE_KEYS: tuple[str, ...] = ("input_tokens", "output_tokens", "total_tokens")
EMBEDDING_USAGE_KEYS: tuple[str, ...] = ("prompt_tokens", "total_tokens")


def build_response_result(
    *,
    item_id: str,
    question: str,
    ground_truth: str,
    question_id: Any,
    generated_output: str,
    failed: bool,
    response_id: str | None = None,
    usage: dict[str, int] | None = None,
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One Stage-1 per-item result, in the batch path's shape."""
    return {
        "item_id": item_id,
        "question": question,
        "generated_output": generated_output,
        "ground_truth": ground_truth,
        "response_id": response_id,
        "usage": usage,
        "question_id": question_id,
        "failed": failed,
        "retrieved_chunks": retrieved_chunks,
    }


def build_embedding_failure(item_id: str, error: str) -> dict[str, Any]:
    """One failed Stage-2 per-pair result."""
    return {
        "item_id": item_id,
        "output_embedding": None,
        "ground_truth_embedding": None,
        "usage": None,
        "failed": True,
        "error": error,
    }


def extract_usage(usage_obj: Any, keys: tuple[str, ...]) -> dict[str, int]:
    """Read the given token counters off an OpenAI usage object, defaulting to 0."""
    return {key: int(field_value(usage_obj, key, 0) or 0) for key in keys}


def parse_embedding_pair(*, item_id: str, response: Any) -> dict[str, Any]:
    """Unpack an embeddings response into one Stage-2 per-pair result.

    The request embeds `[output_text, ground_truth]`, so index 0 is the generated
    output's vector and index 1 the ground truth's.
    """
    data = field_value(response, "data") or []
    if len(data) < 2:
        return build_embedding_failure(
            item_id, f"expected 2 embeddings, got {len(data)}"
        )

    output_embedding: list[float] | None = None
    ground_truth_embedding: list[float] | None = None
    for embedding in data:
        index = field_value(embedding, "index")
        vector = field_value(embedding, "embedding")
        if index == 0:
            output_embedding = vector
        elif index == 1:
            ground_truth_embedding = vector

    return {
        "item_id": item_id,
        "output_embedding": output_embedding,
        "ground_truth_embedding": ground_truth_embedding,
        "usage": extract_usage(field_value(response, "usage"), EMBEDDING_USAGE_KEYS),
        "failed": output_embedding is None or ground_truth_embedding is None,
    }


def sum_usage(results: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    """Sum the per-item `usage` token counts across results, for the given keys."""
    totals = dict.fromkeys(keys, 0)
    for result in results:
        usage = result.get("usage") or {}
        for key in keys:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def is_failure_threshold_breached(*, failed_rows: int, total_rows: int) -> bool:
    """True if the failed-row fraction exceeds EVAL_FAST_FAILURE_THRESHOLD."""
    if total_rows == 0:
        return False
    return (failed_rows / total_rows) > settings.EVAL_FAST_FAILURE_THRESHOLD
