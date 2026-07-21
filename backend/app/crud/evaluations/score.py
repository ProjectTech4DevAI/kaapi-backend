"""
Type definitions for evaluation scores.

This module contains TypedDict definitions for type-safe score data
used throughout the evaluation system.
"""

from typing import NotRequired, TypedDict

DEFAULT_CATEGORY: str = "Other"

# Canonical name/comment for the cosine-similarity score, centralized to avoid
# import cycles.
COSINE_SCORE_NAME: str = "Cosine Similarity"
COSINE_SCORE_COMMENT: str = (
    "Cosine similarity between generated output and ground truth embeddings"
)

GROUND_TRUTH_SCORE_NAME: str = "Adherence to Ground Truth"
KNOWLEDGE_BASE_SCORE_NAME: str = "Adherence to Knowledge Base"
JUDGE_FAILED_REASON: str = "judge_failed"

# Reasons an item cannot be scored, recorded in EvaluationRun.unscoreable.
# "missing_trace_id" appears only in build_embedding_jsonl's internal skipped list.
UNSCOREABLE_REASONS: tuple[str, ...] = (
    "empty_output",
    "empty_ground_truth",
    "embedding_failed",
    "missing_trace_id",
    JUDGE_FAILED_REASON,
)

JUDGE_SYSTEM_PREAMBLE: str = (
    "You are a strict, impartial evaluator. You score an assistant's answer on the "
    "independent metrics listed below in a single pass. Each metric is a float in "
    "[0.0, 1.0] with one or two sentences of reasoning. The metrics are independent "
    "— judge each only against its own inputs and rules; do not let one metric's "
    "verdict bleed into another."
)

GROUND_TRUTH_JUDGE_PROMPT: str = (
    'Adherence to Ground Truth (score key "ground_truth"):\n'
    "Judge ONLY whether the assistant's answer conveys the same correct information "
    "as the golden answer.\n"
    "- Judge meaning, not wording. A correct paraphrase, a different order, or extra "
    "detail that is also correct must score high.\n"
    "- Lower the score for information that is missing, incomplete, or contradicts "
    "the golden answer. An answer that states something the golden answer does not, "
    "and that would be wrong, is a factual error.\n"
    "- Do NOT reward or penalize style, tone, length, or language.\n"
    "- Do NOT use any outside knowledge; the golden answer is the source of truth.\n"
    "- Do NOT answer the question yourself.\n"
    "Reasoning: name what was correct or what was missing/contradicted."
)

KNOWLEDGE_BASE_JUDGE_PROMPT: str = (
    'Adherence to Knowledge Base (score key "knowledge_base"):\n'
    "Judge ONLY whether the answer's claims are supported by the retrieved "
    "knowledge-base chunks (groundedness / hallucination detection).\n"
    "- Break the answer into its distinct factual claims. A claim is supported iff "
    "it is stated in, or directly entailed by, at least one chunk.\n"
    "- Score = supported claims / total claims: all supported = 1.0; mostly "
    "unsupported trends toward 0.0.\n"
    "- Generic pleasantries or a safe refusal that make no factual claim are "
    "trivially grounded — score them high.\n"
    "- Judge groundedness ONLY, not correctness, completeness, or "
    "instruction-following. A claim faithful to the chunks is grounded even if the "
    "chunks are themselves wrong.\n"
    "- Do NOT use any outside knowledge; the retrieved chunks are the ONLY allowed "
    "source of support.\n"
    "Reasoning: when the score is below 1.0, name the specific unsupported or "
    "invented claim."
)

JUDGE_OUTPUT_INSTRUCTION: str = (
    "Respond with ONLY a single JSON object mapping each metric key to its result, of "
    'the form {{"<metric_key>": {{"score": <float between 0 and 1>, "reasoning": '
    '"<one or two sentences>"}}}}. Include exactly these metric keys: {metric_keys}. '
    "Output nothing else."
)


class TraceScore(TypedDict):
    """A score attached to a trace."""

    name: str
    value: float | str
    data_type: str
    comment: NotRequired[str]
    # True for placeholder scores on unscoreable items; excluded from summary stats.
    unscoreable: NotRequired[bool]


class TraceData(TypedDict):
    """Data for a single trace including Q&A and scores."""

    trace_id: str
    question: str
    llm_answer: str
    question_id: int | None
    ground_truth_answer: str
    category: str
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


class EvaluationScore(TypedDict):
    """Complete evaluation score data with traces and summary statistics."""

    summary_scores: list[SummaryScore]
    traces: list[TraceData]
