"""Per-trace record building for fast evaluation runs.

Pure: turns the responses unit plus the scores already computed for it into the
`TraceData` records the read path serves. Keyed by `ref` (trace_id when the run
is traced, else item_id) so untraced v2 runs persist the same shape.
"""

from typing import Any

from app.crud.evaluations.judge import JudgeMetricEnum, JudgeMetricSpec, JudgeResult
from app.crud.evaluations.score import (
    COSINE_SCORE_COMMENT,
    COSINE_SCORE_NAME,
    DEFAULT_CATEGORY,
    JUDGE_FAILED_REASON,
    TraceData,
    TraceScore,
    verdict_from_score,
)

# How many top KB matches to name in the knowledge_base trace comment.
_KB_TOP_CHUNKS = 3

_KB_NOT_QUERIED = "Knowledge base not queried."
_KB_SCORE_UNAVAILABLE = "Knowledge base score unavailable for this row."


def format_top_kb_matches(sorted_chunks: list[dict[str, Any]]) -> str:
    """Top-N retrieved chunks as 'biu-1.pdf (90.6%), faq.pdf (66.3%)'.

    Expects chunks pre-sorted by score desc; old S3 payloads may lack filename.
    """
    matches = [
        f"{c.get('filename') or 'unknown'} ({c.get('score', 0) * 100:.1f}%)"
        for c in sorted_chunks
    ]
    return ", ".join(matches[:_KB_TOP_CHUNKS])


def _cosine_trace_score(
    *, cosine: float | None, unscoreable_reason: str | None
) -> TraceScore | None:
    """The cosine entry for one v1 trace, or None when neither applies."""
    if cosine is not None:
        return {
            "name": COSINE_SCORE_NAME,
            "value": round(cosine, 2),
            "data_type": "NUMERIC",
            "comment": COSINE_SCORE_COMMENT,
        }
    # A judge_failed-only reason is about the judge, not cosine, so it gets no
    # placeholder. Other reasons get a 0 excluded from summary stats by the marker.
    if unscoreable_reason is not None and unscoreable_reason != JUDGE_FAILED_REASON:
        return {
            "name": COSINE_SCORE_NAME,
            "value": 0,
            "data_type": "NUMERIC",
            "comment": f"Cannot compute: {unscoreable_reason}",
            "unscoreable": True,
        }
    return None


def _kb_placeholder_score(
    *, spec: JudgeMetricSpec, sorted_chunks: list[dict[str, Any]]
) -> TraceScore:
    """Human-readable N/A for a row the knowledge_base metric was dropped on."""
    # ponytail: empty chunks under auto tool_choice ~= not queried; a "was queried"
    # flag would disambiguate an empty-store hit, not worth plumbing.
    reason = _KB_NOT_QUERIED if not sorted_chunks else _KB_SCORE_UNAVAILABLE
    return {
        "name": spec.score_name,
        "value": "N/A",
        "data_type": "CATEGORICAL",
        "comment": reason,
        "unscoreable": True,
    }


def _judge_trace_scores(
    *,
    judge_result: JudgeResult,
    metrics: list[JudgeMetricSpec],
    retrieved_chunks: list[dict[str, Any]],
) -> list[TraceScore]:
    """One row's judge entries, each carrying its reasoning as the score comment."""
    sorted_chunks = sorted(
        retrieved_chunks, key=lambda c: c.get("score", 0), reverse=True
    )
    top_matches = format_top_kb_matches(sorted_chunks)

    scores: list[TraceScore] = []
    for spec in metrics:
        metric_score = judge_result.metrics.get(spec.key)
        is_kb = spec.key == JudgeMetricEnum.KNOWLEDGE_BASE
        if metric_score is None:
            if is_kb:
                scores.append(
                    _kb_placeholder_score(spec=spec, sorted_chunks=sorted_chunks)
                )
            continue

        comment = metric_score.reasoning
        if is_kb:
            comment = f"{comment} | Top matches: {top_matches}"
        rounded_score = round(metric_score.score, 2)
        scores.append(
            {
                "name": spec.score_name,
                "value": rounded_score,
                "data_type": "NUMERIC",
                "comment": comment,
                "verdict": verdict_from_score(rounded_score),
            }
        )
    return scores


def build_trace_records(
    *,
    response_results: list[dict[str, Any]],
    item_refs: dict[str, str],
    is_judge_run: bool | None,
    judge_results: dict[str, JudgeResult],
    metrics: list[JudgeMetricSpec],
    cosine_by_item_id: dict[str, float],
    unscoreable: dict[str, str],
) -> list[TraceData]:
    """Build every trace record for the run.

    v1 rows carry the cosine score (or its unscoreable placeholder); v2 rows carry
    one entry per judge metric and no cosine placeholder.
    """
    traces: list[TraceData] = []
    for response in response_results:
        item_id: str = response["item_id"]
        ref: str = item_refs.get(item_id) or item_id
        trace_scores: list[TraceScore] = []

        if not is_judge_run:
            cosine_score = _cosine_trace_score(
                cosine=cosine_by_item_id.get(item_id),
                unscoreable_reason=unscoreable.get(ref),
            )
            if cosine_score is not None:
                trace_scores.append(cosine_score)

        judge_result = judge_results.get(item_id)
        if judge_result is not None:
            trace_scores.extend(
                _judge_trace_scores(
                    judge_result=judge_result,
                    metrics=metrics,
                    retrieved_chunks=response.get("retrieved_chunks") or [],
                )
            )

        trace: TraceData = {
            "trace_id": ref,
            "question": response.get("question", ""),
            "llm_answer": response.get("generated_output", ""),
            "ground_truth_answer": response.get("ground_truth", ""),
            "question_id": response.get("question_id"),
            "category": response.get("category") or DEFAULT_CATEGORY,
            "scores": trace_scores,
        }
        traces.append(trace)
    return traces
