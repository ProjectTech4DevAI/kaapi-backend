"""
Step-forward merge helpers for evaluation trace scores.

A Langfuse resync may return fewer traces than a previous sync (transient
per-trace fetch failures, eventual consistency on scores still being written,
etc.). Blindly overwriting the cached result with the freshly fetched one can
therefore lose data and make the visible pair count go *down* (e.g. 29/30 -> 27/30).

These helpers make resync monotonic ("always a step forward"): the cached
traces are merged with the freshly fetched ones by ``trace_id`` so that a trace
we already have is never dropped, and summary statistics are recomputed from the
merged union so the totals stay consistent with the traces.
"""

import logging
from typing import Any

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
    Recompute summary statistics from a list of traces.

    Aggregates every per-trace score by name. Numeric scores get avg/std,
    categorical scores get a value distribution. ``total_pairs`` is the number of
    traces that carry a non-null value for that score.

    Args:
        traces: Traces to aggregate (typically the merged union of cached + fresh).

    Returns:
        Summary scores derived from ``traces``.
    """
    # Track aggregations by score name: {name: {"data_type": str, "values": list}}
    score_aggregations: dict[str, dict[str, Any]] = {}

    for trace in traces:
        for score_entry in trace.get("scores", []):
            score_name = score_entry["name"]
            score_value = score_entry["value"]
            data_type = score_entry.get("data_type") or "NUMERIC"
            if score_value is None:
                continue
            if score_name not in score_aggregations:
                score_aggregations[score_name] = {
                    "data_type": data_type,
                    "values": [],
                }
            score_aggregations[score_name]["values"].append(score_value)

    summary_scores: list[SummaryScore] = []
    for score_name, agg_data in score_aggregations.items():
        data_type = agg_data["data_type"]
        values = agg_data["values"]

        if data_type == "CATEGORICAL":
            distribution: dict[str, int] = {}
            for val in values:
                str_val = str(val)
                distribution[str_val] = distribution.get(str_val, 0) + 1
            summary_scores.append(
                {
                    "name": score_name,
                    "distribution": distribution,
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
    Merge two versions of the same trace (same ``trace_id``).

    Scores are unioned by name; the freshly fetched value wins on conflict (it is
    the more up-to-date one) and any score present only in one version is kept.
    Text fields prefer the fresh value when it is non-empty, otherwise the cached
    one — so we never regress a populated field back to empty.
    """
    merged_scores_by_name: dict[str, TraceScore] = {
        s["name"]: s for s in existing.get("scores", [])
    }
    for s in fresh.get("scores", []):
        merged_scores_by_name[s["name"]] = s

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


def merge_trace_data(
    existing_traces: list[TraceData],
    fresh_traces: list[TraceData],
) -> tuple[list[TraceData], dict[str, int]]:
    """
    Step-forward merge of cached traces with freshly fetched ones.

    Union by ``trace_id``:
    - present only in the cache (Langfuse failed to return it this time) -> kept
    - present only in the fresh fetch (newly available) -> added
    - present in both -> merged via :func:`_merge_single_trace`

    This guarantees the merged result is never smaller than the cache, so repeated
    resyncs can only grow the pair count (e.g. 15/30 -> 24/30 -> 30/30).

    Args:
        existing_traces: Previously cached traces (may be empty on first sync).
        fresh_traces: Traces fetched from Langfuse in this resync.

    Returns:
        Tuple of (merged traces, stats) where stats counts ``reused`` (cache kept
        as-is), ``updated`` (cache enriched with fresh data), and ``added`` (new).
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
        existing = existing_by_id.get(trace_id)
        fresh = fresh_by_id.get(trace_id)

        if existing and not fresh:
            merged.append(existing)
            stats["reused"] += 1
        elif fresh and not existing:
            merged.append(fresh)
            stats["added"] += 1
        else:
            merged_trace = _merge_single_trace(existing, fresh)
            merged.append(merged_trace)
            if merged_trace == existing:
                stats["reused"] += 1
            else:
                stats["updated"] += 1

    return merged, stats


def merge_scores_step_forward(
    existing_score: EvaluationScore | None,
    fresh_score: EvaluationScore,
) -> tuple[EvaluationScore, dict[str, int]]:
    """
    Merge a freshly fetched evaluation score into the cached one, monotonically.

    Traces are merged step-forward (see :func:`merge_trace_data`) and summary
    statistics are recomputed from the merged traces so the totals match. Any
    summary score that has no per-trace representation (e.g. a summary-only metric)
    is preserved from the existing score.

    Args:
        existing_score: Previously cached score (``None``/empty on first sync).
        fresh_score: Score freshly fetched from Langfuse.

    Returns:
        Tuple of (merged EvaluationScore, merge stats).
    """
    existing_traces = (existing_score or {}).get("traces", []) or []
    fresh_traces = fresh_score.get("traces", []) or []

    merged_traces, stats = merge_trace_data(existing_traces, fresh_traces)

    # Recompute summaries from the union so total_pairs reflects the merged set.
    recomputed_summary = compute_summary_scores(merged_traces)

    # Preserve any summary-only score (no per-trace value) from the existing score;
    # recomputed summaries (derived from the merged traces) take precedence by name.
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
