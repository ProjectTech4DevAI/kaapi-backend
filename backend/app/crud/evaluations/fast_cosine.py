"""Cosine scoring for v1 fast runs.

Pure: takes the responses + embeddings units and returns everything Stage 3
persists for a non-judge run. v2 judged runs never embed, so they never reach
this module.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.crud.evaluations.embeddings import calculate_cosine_similarity
from app.crud.evaluations.merge import apply_cosine_breakdown
from app.crud.evaluations.score import (
    COSINE_SCORE_NAME,
    UNSCOREABLE_EMBEDDING_FAILED,
    UNSCOREABLE_EMPTY_GROUND_TRUTH,
    UNSCOREABLE_EMPTY_OUTPUT,
    SummaryScore,
)

_PER_ITEM_SCORE_PRECISION = 6


def classify_empty_side(response: dict[str, Any]) -> str | None:
    """Why a row can't be scored from its own text, or None if both sides are present."""
    if not response.get("generated_output"):
        return UNSCOREABLE_EMPTY_OUTPUT
    if not response.get("ground_truth"):
        return UNSCOREABLE_EMPTY_GROUND_TRUTH
    return None


def build_item_refs(
    response_results: list[dict[str, Any]], trace_id_mapping: dict[str, str]
) -> dict[str, str]:
    """Map each item id to the key its scores hang off: trace_id when traced, else item id."""
    return {
        response["item_id"]: trace_id_mapping.get(response["item_id"])
        or response["item_id"]
        for response in response_results
    }


@dataclass
class CosineScoring:
    """Everything Stage 3 persists for a v1 cosine run."""

    item_id_to_score: dict[str, float] = field(default_factory=dict)
    per_item_scores: dict[str, float] = field(default_factory=dict)
    unscoreable: dict[str, str] = field(default_factory=dict)
    write_items: list[dict[str, Any]] = field(default_factory=list)
    summary_scores: list[SummaryScore] = field(default_factory=list)


def _has_embedding_pair(pair: dict[str, Any] | None) -> bool:
    return (
        pair is not None
        and pair.get("output_embedding") is not None
        and pair.get("ground_truth_embedding") is not None
    )


def _summary_score(
    similarities: list[float], *, total_items: int, unscoreable: dict[str, str]
) -> list[SummaryScore]:
    if similarities:
        array = np.array(similarities)
        avg, std = float(np.mean(array)), float(np.std(array))
    else:
        avg = std = 0.0

    return apply_cosine_breakdown(
        [
            {
                "name": COSINE_SCORE_NAME,
                "avg": round(avg, 2),
                "std": round(std, 2),
                "total_pairs": len(similarities),
                "data_type": "NUMERIC",
            }
        ],
        total_items=total_items,
        unscoreable=unscoreable or None,
    )


def score_cosine_run(
    *,
    response_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]] | None,
    item_refs: dict[str, str],
    trace_id_mapping: dict[str, str],
    total_items: int,
) -> CosineScoring:
    """Score every row by cosine similarity and build the Langfuse write list.

    A row without a usable embedding pair is recorded in `unscoreable` with the
    reason the UI shows, and stays out of the average.
    """
    pair_by_item_id = {
        r["item_id"]: r for r in (embedding_results or []) if not r.get("failed")
    }

    result = CosineScoring()
    similarities: list[float] = []
    scored_writes: list[dict[str, Any]] = []

    for response in response_results:
        item_id = response["item_id"]
        ref = item_refs[item_id]
        pair = pair_by_item_id.get(item_id)
        if not _has_embedding_pair(pair):
            result.unscoreable[ref] = (
                classify_empty_side(response) or UNSCOREABLE_EMBEDDING_FAILED
            )
            continue

        assert pair is not None  # guaranteed by _has_embedding_pair
        cosine = calculate_cosine_similarity(
            pair["output_embedding"], pair["ground_truth_embedding"]
        )
        similarities.append(cosine)
        result.item_id_to_score[item_id] = cosine
        scored_writes.append(
            {"trace_id": trace_id_mapping.get(item_id), "cosine_similarity": cosine}
        )

    unscoreable_writes = [
        {"trace_id": trace_id_mapping[item_id], "unscoreable": True, "reason": reason}
        for item_id, ref in item_refs.items()
        if item_id in trace_id_mapping and (reason := result.unscoreable.get(ref))
    ]
    # Untraced runs have no trace_id to write against, so those entries drop out.
    result.write_items = [
        w for w in scored_writes if w["trace_id"] is not None
    ] + unscoreable_writes

    result.per_item_scores = {
        item_refs[item_id]: round(float(score), _PER_ITEM_SCORE_PRECISION)
        for item_id, score in result.item_id_to_score.items()
    }
    result.summary_scores = _summary_score(
        similarities, total_items=total_items, unscoreable=result.unscoreable
    )
    return result
