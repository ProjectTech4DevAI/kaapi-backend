"""
Step-forward merge helpers for evaluation trace scores.

A Langfuse resync can return fewer traces than before (transient fetch failures,
scores not yet written). Merging by ``trace_id`` instead of overwriting keeps the
result monotonic, so the pair count can only grow across resyncs (never 29 -> 27).
"""

import itertools
import logging
from collections import Counter

import numpy as np

from app.crud.evaluations.score import (
    EvaluationScore,
    SummaryScore,
    TraceData,
    TraceScore,
)

logger = logging.getLogger(__name__)


def compute_summary_scores(traces: list[TraceData]) -> list[SummaryScore]:
    """
    Aggregate per-trace scores by name: numeric scores get avg/std, categorical
    scores get a value distribution. ``total_pairs`` counts non-null values.
    """
    # {name: {"data_type": str, "values": list}}
    score_aggregations: dict[str, dict] = {}
    all_scores = itertools.chain.from_iterable(t.get("scores", []) for t in traces)
    for entry in all_scores:
        if entry["value"] is None:
            continue
        agg = score_aggregations.setdefault(
            entry["name"],
            {"data_type": entry.get("data_type") or "NUMERIC", "values": []},
        )
        agg["values"].append(entry["value"])

    summary_scores: list[SummaryScore] = []
    for score_name, agg_data in score_aggregations.items():
        data_type = agg_data["data_type"]
        values = agg_data["values"]

        if data_type == "CATEGORICAL":
            summary_scores.append(
                {
                    "name": score_name,
                    "distribution": dict(Counter(str(v) for v in values)),
                    "total_pairs": len(values),
                    "data_type": data_type,
                }
            )
        else:
            numeric_values = [float(v) for v in values]
            summary_scores.append(
                {
                    "name": score_name,
                    "avg": round(float(np.mean(numeric_values)), 2),
                    "std": round(float(np.std(numeric_values)), 2),
                    "total_pairs": len(numeric_values),
                    "data_type": data_type,
                }
            )

    return summary_scores


def _merge_single_trace(existing: TraceData, fresh: TraceData) -> TraceData:
    """
    Merge two versions of the same trace. Scores are unioned by name (fresh wins
    on conflict); text fields prefer the fresh value when non-empty, else cached.
    """
    merged_scores_by_name: dict[str, TraceScore] = {
        score["name"]: score for score in existing.get("scores", [])
    }
    for fresh_score in fresh.get("scores", []):
        merged_scores_by_name[fresh_score["name"]] = fresh_score

    return {
        "trace_id": fresh.get("trace_id") or existing.get("trace_id", ""),
        "question": fresh.get("question") or existing.get("question", ""),
        "llm_answer": fresh.get("llm_answer") or existing.get("llm_answer", ""),
        "ground_truth_answer": (
            fresh.get("ground_truth_answer") or existing.get("ground_truth_answer", "")
        ),
        "question_id": fresh.get("question_id") or existing.get("question_id"),
        "scores": list(merged_scores_by_name.values()),
    }


def _reconcile_trace(
    existing: TraceData | None,
    fresh: TraceData | None,
) -> tuple[TraceData, str]:
    """
    Pick the winning version of one trace and classify it as ``added``, ``reused``,
    or ``updated``. Exactly one of ``existing``/``fresh`` may be None, never both.
    """
    if existing is None:
        return fresh, "added"
    if fresh is None:
        return existing, "reused"
    merged = _merge_single_trace(existing, fresh)
    return merged, "reused" if merged == existing else "updated"


def merge_trace_data(
    existing_traces: list[TraceData],
    fresh_traces: list[TraceData],
) -> tuple[list[TraceData], dict[str, int]]:
    """
    Union cached and fresh traces by ``trace_id`` (a trace only in the cache is
    kept, never dropped), so the result is never smaller than the cache. Returns
    the merged traces and a ``{reused, updated, added}`` count.
    """
    existing_by_id: dict[str, TraceData] = {t["trace_id"]: t for t in existing_traces}
    fresh_by_id: dict[str, TraceData] = {t["trace_id"]: t for t in fresh_traces}

    # Preserve cache order first, then append genuinely new traces.
    ordered_ids = list(existing_by_id.keys()) + [
        tid for tid in fresh_by_id if tid not in existing_by_id
    ]

    merged: list[TraceData] = []
    stats = {"reused": 0, "updated": 0, "added": 0}

    for trace_id in ordered_ids:
        trace, category = _reconcile_trace(
            existing_by_id.get(trace_id), fresh_by_id.get(trace_id)
        )
        merged.append(trace)
        stats[category] += 1

    return merged, stats


def merge_scores_step_forward(
    existing_score: EvaluationScore | None,
    fresh_score: EvaluationScore,
) -> tuple[EvaluationScore, dict[str, int]]:
    """
    Merge a freshly fetched score into the cached one monotonically: traces are
    merged step-forward and summaries recomputed from the union. Summary-only
    scores (no per-trace value) are preserved from the existing score.
    """
    existing_traces = (existing_score or {}).get("traces", []) or []
    fresh_traces = fresh_score.get("traces", []) or []

    merged_traces, stats = merge_trace_data(existing_traces, fresh_traces)
    recomputed_summary = compute_summary_scores(merged_traces)

    # Recomputed summaries take precedence; summary-only scores survive.
    summary_by_name: dict[str, SummaryScore] = {
        s["name"]: s for s in (existing_score or {}).get("summary_scores", [])
    }
    for s in recomputed_summary:
        summary_by_name[s["name"]] = s

    merged: EvaluationScore = {
        "summary_scores": list(summary_by_name.values()),
        "traces": merged_traces,
    }
    return merged, stats
