"""Score types, verdict banding and the run-level overall rollup."""

from enum import Enum
from typing import NotRequired, TypedDict

# Rubric text lives in judge_prompts; re-exported for existing importers.
from app.crud.evaluations.judge_prompts import (  # noqa: F401
    GROUND_TRUTH_JUDGE_PROMPT,
    JUDGE_OUTPUT_INSTRUCTION,
    JUDGE_SYSTEM_PREAMBLE,
    KNOWLEDGE_BASE_JUDGE_PROMPT,
    PROMPT_JUDGE_PROMPT,
)

DEFAULT_CATEGORY: str = "Other"


class VerdictEnum(str, Enum):
    """Qualitative band derived from a 0–5 judge-metric score."""

    NEEDS_IMPROVEMENT = "Needs Improvement"
    NEEDS_REFINEMENT = "Needs Refinement"
    GOOD = "Good"


VERDICT_NEEDS_IMPROVEMENT_BELOW: float = 2.0
VERDICT_GOOD_AT_OR_ABOVE: float = 4.0


def verdict_from_score(score: float) -> VerdictEnum:
    """Map a 0–5 judge-metric score to its verdict band.

    Boundaries: below 2 → Needs Improvement, 2 to <4 → Needs Refinement,
    4 and above → Good.
    """
    if score < VERDICT_NEEDS_IMPROVEMENT_BELOW:
        return VerdictEnum.NEEDS_IMPROVEMENT
    if score < VERDICT_GOOD_AT_OR_ABOVE:
        return VerdictEnum.NEEDS_REFINEMENT
    return VerdictEnum.GOOD


# Canonical name/comment for the cosine-similarity score, centralized to avoid
# import cycles.
COSINE_SCORE_NAME: str = "Cosine Similarity"
COSINE_SCORE_COMMENT: str = (
    "Cosine similarity between generated output and ground truth embeddings"
)

GROUND_TRUTH_SCORE_NAME: str = "Adherence to Ground Truth"
PROMPT_SCORE_NAME: str = "Adherence to Prompt"
KNOWLEDGE_BASE_SCORE_NAME: str = "Adherence to Knowledge Base"
# Reasons an item cannot be scored, recorded in EvaluationRun.unscoreable.
# MISSING_TRACE_ID appears only in build_embedding_jsonl's internal skipped list;
# EMBEDDING_FAILED is v1-only (cosine), since v2 judged runs never embed.
UNSCOREABLE_EMPTY_OUTPUT: str = "empty_output"
UNSCOREABLE_EMPTY_GROUND_TRUTH: str = "empty_ground_truth"
UNSCOREABLE_EMBEDDING_FAILED: str = "embedding_failed"
UNSCOREABLE_MISSING_TRACE_ID: str = "missing_trace_id"
JUDGE_FAILED_REASON: str = "judge_failed"

UNSCOREABLE_REASONS: tuple[str, ...] = (
    UNSCOREABLE_EMPTY_OUTPUT,
    UNSCOREABLE_EMPTY_GROUND_TRUTH,
    UNSCOREABLE_EMBEDDING_FAILED,
    UNSCOREABLE_MISSING_TRACE_ID,
    JUDGE_FAILED_REASON,
)


class TraceScore(TypedDict):
    """A score attached to a trace."""

    name: str
    value: float | str
    data_type: str
    comment: NotRequired[str]
    verdict: NotRequired[str]
    # True for placeholder scores on unscoreable items; excluded from summary stats.
    unscoreable: NotRequired[bool]


class TraceData(TypedDict):
    """Data for a single trace including Q&A and scores."""

    trace_id: str
    question: str
    llm_answer: str
    question_id: int | None
    ground_truth_answer: str
    category: NotRequired[str]
    scores: list[TraceScore]


class CategoryMetrics(TypedDict):
    """Aggregated per-category metrics across an eval run.

    `avg_cosine` and `avg_correctness` are the simple arithmetic means of the
    cosine-similarity and correctness scores for traces in this category; null
    when the category has no traces with that score.
    """

    category: str
    total_evals: int
    avg_cosine: float | None
    avg_correctness: float | None


class NumericSummaryScore(TypedDict):
    """Summary statistics for a numeric score across all traces."""

    name: str
    avg: float
    std: float
    total_pairs: int
    data_type: str
    # UI denominator (total dataset items) and per-reason unscoreable breakdown.
    # Present on the cosine-similarity score.
    total_items: NotRequired[int]
    unscoreable: NotRequired[dict[str, int]]


class CategoricalSummaryScore(TypedDict):
    """Summary statistics for a categorical score across all traces."""

    name: str
    distribution: dict[str, int]
    total_pairs: int
    data_type: str


SummaryScore = NumericSummaryScore | CategoricalSummaryScore


class OverallDimension(TypedDict):
    """One judge metric's contribution to the run-level overall score."""

    name: str
    key: str
    score: float
    weight: float
    delta: float
    verdict: str


class OverallSummary(TypedDict):
    """Run-level weighted quality view for a v2 judge run.

    `ai_summary` is filled by a best-effort LLM step after the deterministic
    fields; it stays None when no summary was generated.
    """

    overall_score: float
    verdict: str
    ai_summary: str | None
    breakdown: list[OverallDimension]


def compute_overall_summary(
    *,
    metric_avgs: dict[str, float],
    metric_weights: dict[str, float],
    metric_names: dict[str, str],
) -> OverallSummary | None:
    """Weighted run-level overall score + per-dimension breakdown. No LLM.

    All three dicts are keyed by metric key value. Only metrics present in
    `metric_avgs` (i.e. that actually scored ≥1 row) count; a metric with no
    scoreable rows is dropped and the remaining base weights are renormalized to
    sum to 1, so a missing metric never drags the overall down. Returns None when
    nothing scored. `ai_summary` is None here — the LLM step fills it later.
    """
    scored_keys = [key for key in metric_avgs if key in metric_weights]
    weight_total = sum(metric_weights[key] for key in scored_keys)
    if not scored_keys or weight_total <= 0:
        return None

    renorm_weights = {key: metric_weights[key] / weight_total for key in scored_keys}
    # Round the overall once, then reuse it everywhere so the badge and the number
    # (and every delta) are computed from the same value and can never disagree.
    overall_score = round(
        sum(renorm_weights[key] * metric_avgs[key] for key in scored_keys), 2
    )
    verdict = verdict_from_score(overall_score).value

    breakdown: list[OverallDimension] = []
    for key in scored_keys:
        avg = round(metric_avgs[key], 2)
        breakdown.append(
            {
                "name": metric_names.get(key, key),
                "key": key,
                "score": avg,
                "weight": round(renorm_weights[key], 2),
                "delta": round(avg - overall_score, 2),
                "verdict": verdict_from_score(avg).value,
            }
        )

    return {
        "overall_score": overall_score,
        "verdict": verdict,
        "ai_summary": None,
        "breakdown": breakdown,
    }


class EvaluationScore(TypedDict):
    """Complete evaluation score data with traces and summary statistics.

    `traces` is omitted from the DB-persisted summary-only variant (when
    per-trace records are uploaded to S3 instead), so it's optional here.
    """

    summary_scores: list[SummaryScore]
    traces: NotRequired[list[TraceData]]
    overall: NotRequired[OverallSummary]
    category_metrics: NotRequired[list[CategoryMetrics]]
