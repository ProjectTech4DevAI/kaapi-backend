"""
Step-forward merge helpers for evaluation trace scores.

A Langfuse resync can return fewer traces than before (transient fetch failures,
scores not yet written). Merging by ``trace_id`` instead of overwriting keeps the
result monotonic, so the pair count can only grow across resyncs (never 29 -> 27).
"""

import itertools
import logging
from collections import Counter
from typing import Any

import numpy as np

from app.crud.evaluations.score import (
    COSINE_SCORE_COMMENT,
    COSINE_SCORE_NAME,
    DEFAULT_CATEGORY,
    EvaluationScore,
    SummaryScore,
    TraceData,
    TraceScore,
    UNSCOREABLE_REASONS,
)

logger = logging.getLogger(__name__)


def sort_traces_by_question_id(traces: list[TraceData]) -> list[TraceData]:
    """
    Return ``traces`` ordered by the 1-based ``question_id`` assigned at Langfuse
    upload time (effectively the CSV row index). Without this, traces come back in
    ThreadPoolExecutor completion order, so the API response sequence is a race
    rather than the CSV's natural order. Traces missing or with non-numeric
    question_id are pushed to the end so the sort is total even for legacy traces.
    """

    def _key(trace: TraceData) -> tuple[int, int]:
        qid = trace.get("question_id")
        if isinstance(qid, int):
            return (0, qid)
        if isinstance(qid, str) and qid.strip().isdigit():
            return (0, int(qid))
        return (1, 0)

    return sorted(traces, key=_key)


def compute_summary_scores(traces: list[TraceData]) -> list[SummaryScore]:
    """
    Aggregate per-trace scores by name: numeric scores get avg/std, categorical
    scores get a value distribution. ``total_pairs`` counts non-null values.

    Entries flagged ``unscoreable`` (placeholder 0-scores written to explain a
    gap, e.g. empty model output) are skipped so they never drag the average
    down or inflate ``total_pairs`` — they exist only for per-trace display.
    """
    # {name: {"data_type": str, "values": list}}
    score_aggregations: dict[str, dict] = {}
    all_scores = itertools.chain.from_iterable(t.get("scores", []) for t in traces)
    for entry in all_scores:
        if entry["value"] is None:
            continue
        if entry.get("unscoreable"):
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


def summarize_unscoreable(
    unscoreable: dict[str, str] | None,
) -> dict[str, int]:
    """Count unscoreable items per reason, e.g. ``{"empty_output": 3}``.

    ``unscoreable`` is the ``EvaluationRun.unscoreable`` map ``{trace_id: reason}``.
    Only the known reasons in ``UNSCOREABLE_REASONS`` are counted; unknown
    reasons are bucketed under ``"other"`` so nothing is silently dropped.
    """
    breakdown: dict[str, int] = {}
    for reason in (unscoreable or {}).values():
        key = reason if reason in UNSCOREABLE_REASONS else "other"
        breakdown[key] = breakdown.get(key, 0) + 1
    return breakdown


def apply_cosine_breakdown(
    summary_scores: list[SummaryScore],
    *,
    total_items: int | None,
    unscoreable: dict[str, str] | None,
) -> list[SummaryScore]:
    """Attach ``total_items`` (the UI denominator) and the ``unscoreable``
    per-reason breakdown to the cosine-similarity summary score, in place.

    ``compute_summary_scores`` rebuilds summaries from traces alone and so
    cannot know the run-level denominator/flags; this re-applies them wherever
    the summary is (re)built. No-op when there is no cosine summary entry.
    """
    breakdown = summarize_unscoreable(unscoreable)
    for entry in summary_scores:
        if entry.get("name") == COSINE_SCORE_NAME:
            if total_items is not None:
                entry["total_items"] = total_items
            entry["unscoreable"] = breakdown
    return summary_scores


def backfill_missing_scores(
    traces: list[TraceData],
    per_item_scores: dict[str, float] | None,
) -> list[TraceData]:
    """Inject durable cosine scores into traces that lack one, in place.

    ``per_item_scores`` is the ``EvaluationRun.per_item_scores`` map
    ``{trace_id: cosine}`` — the durable source of truth for computed scores.
    For any trace whose ``scores`` has no (non-unscoreable) cosine entry but
    whose ``trace_id`` is in the map, append the stored cosine score. This
    recovers scores that were computed but never landed in Langfuse, so the
    resync count reflects everything we actually computed.
    """
    if not per_item_scores:
        return traces

    for trace in traces:
        trace_id = trace.get("trace_id")
        if not trace_id or trace_id not in per_item_scores:
            continue
        has_cosine = any(
            s.get("name") == COSINE_SCORE_NAME and not s.get("unscoreable")
            for s in trace.get("scores", [])
        )
        if has_cosine:
            continue
        trace.setdefault("scores", []).append(
            {
                "name": COSINE_SCORE_NAME,
                "value": round(float(per_item_scores[trace_id]), 2),
                "data_type": "NUMERIC",
                "comment": COSINE_SCORE_COMMENT,
            }
        )
    return traces


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

    merged: dict[str, Any] = {
        "trace_id": fresh.get("trace_id") or existing.get("trace_id", ""),
        "question": fresh.get("question") or existing.get("question", ""),
        "llm_answer": fresh.get("llm_answer") or existing.get("llm_answer", ""),
        "ground_truth_answer": (
            fresh.get("ground_truth_answer") or existing.get("ground_truth_answer", "")
        ),
        "question_id": fresh.get("question_id") or existing.get("question_id"),
        "scores": list(merged_scores_by_name.values()),
    }

    if "category" in existing or "category" in fresh:
        merged["category"] = (
            fresh.get("category") or existing.get("category") or DEFAULT_CATEGORY
        )

    return merged


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
    canonical_existing = _merge_single_trace(existing, existing)
    return merged, "reused" if merged == canonical_existing else "updated"


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
    per_item_scores: dict[str, float] | None = None,
) -> tuple[EvaluationScore, dict[str, int]]:
    """
    Merge a freshly fetched score into the cached one monotonically: traces are
    merged step-forward and summaries recomputed from the union. Summary-only
    scores (no per-trace value) are preserved from the existing score.

    When ``per_item_scores`` (the durable ``{trace_id: cosine}`` source of truth)
    is provided, any merged trace still missing a cosine score is backfilled from
    it before summaries are recomputed — so scores that were computed but never
    landed in Langfuse are recovered, and the count never regresses below what we
    actually computed.
    """
    existing_traces = (existing_score or {}).get("traces", []) or []
    fresh_traces = fresh_score.get("traces", []) or []

    merged_traces, stats = merge_trace_data(existing_traces, fresh_traces)
    merged_traces = backfill_missing_scores(merged_traces, per_item_scores)
    merged_traces = sort_traces_by_question_id(merged_traces)
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
