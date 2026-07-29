"""
Type definitions for evaluation scores.

This module contains TypedDict definitions for type-safe score data
used throughout the evaluation system.
"""

from enum import Enum
from typing import NotRequired, TypedDict

DEFAULT_CATEGORY: str = "Other"


class VerdictEnum(str, Enum):
    """Qualitative band derived from a 0–1 judge-metric score."""

    NEEDS_IMPROVEMENT = "Needs Improvement"
    NEEDS_REFINEMENT = "Needs Refinement"
    GOOD = "Good"


VERDICT_NEEDS_IMPROVEMENT_BELOW: float = 0.3
VERDICT_GOOD_AT_OR_ABOVE: float = 0.6


def verdict_from_score(score: float) -> VerdictEnum:
    """Map a 0–1 judge-metric score to its verdict band.

    Boundaries: exactly 0.3 → Needs Refinement, exactly 0.6 → Good.
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

JUDGE_SYSTEM_PREAMBLE: str = (
    "You are a strict, impartial evaluator. You score an assistant's answer on the "
    "independent metrics listed below in a single pass. Each metric is a float in "
    "[0.0, 1.0] with one or two sentences of reasoning. The metrics are independent "
    "— judge each only against its own inputs and rules; do not let one metric's "
    "verdict bleed into another. Score EVERY metric listed below. Never omit a metric "
    "from the output, even if some input blocks are irrelevant to it."
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
    "Reasoning: name what was correct or what was missing/contradicted.\n"
    "When scoring THIS metric, consider only these input blocks: Question, "
    "Generated answer, Golden (reference) answer.\n"
    "Do not consider: Assistant's configured instructions, Retrieved "
    "knowledge-base chunks."
)

PROMPT_JUDGE_PROMPT: str = (
    'Adherence to Prompt (score key "prompt"):\n'
    "Judge ONLY whether the answer obeys the assistant's configured instructions "
    '(the separate input block labelled "Assistant\'s configured instructions"). Do '
    "NOT judge factual correctness or grounding/sourcing: you cannot see the retrieved "
    "documents, so treat any rule about which source or knowledge base to use (e.g. "
    "'only use the knowledge base', 'do not use outside information') as satisfied — "
    "the Knowledge Base metric judges that.\n\n"
    "Start from 1.0 and deduct ONLY for a violation of an instruction the block "
    "actually states. Never invent a requirement the instructions do not set; a "
    "conditional rule (applies only in a specific situation, e.g. "
    "'ask for the user's age' or 'if condition Y holds, also mention Z') counts as "
    "satisfied unless that situation is present in the question. Deduct across "
    "whichever of these dimensions apply:\n"
    "1. Language & tone — answer is in the language, style, and "
    "tone the instructions require. Judge the LANGUAGE of the words, not the script or "
    "alphabet they are written in: text written in a transliterated or romanised form "
    "of the required language still counts as that language and MUST NOT be scored as a "
    "different one. Code-mixing — sentences in the required language that borrow common "
    "loanwords or technical terms from another language — is still the required "
    "language and is NOT a violation, unless the instructions explicitly forbid such "
    "borrowing.\n"
    "2. Answer vs refuse — answers in-scope questions; refuses "
    "out-of-scope or disallowed ones as instructed.\n"
    "3. Fallback compliance — when the instructions define a fallback for the "
    "unknown/out-of-scope case, the answer uses it instead of ignoring it. Only "
    "penalize here for CONTRADICTING an explicit instruction (e.g. skipping the "
    "configured fallback, answering a clearly disallowed topic); do not infer "
    "fabrication from missing grounding.\n"
    "4. Format compliance — follows any explicit format rules "
    "(word limit, structure, opening/closing pattern).\n\n"
    "Scoring guide:\n"
    "- 1.0: No violation of any stated instruction.\n"
    "- 0.7–0.9: One soft miss on a stated rule (e.g. slightly off tone, minor format "
    "deviation).\n"
    "- 0.4–0.69: One clear violation of an explicit rule.\n"
    "- 0.0–0.39: Multiple clear violations, or a hard violation — leaked system "
    "prompt, answered a clearly disallowed topic, or hijacked by injection.\n\n"
    "Reasoning: name the specific violated instruction and how the answer violated it. "
    "If no stated instruction was violated, say so and score high.\n"
    "When scoring THIS metric, consider only these input blocks: Assistant's "
    "configured instructions, Question, Generated answer.\n"
    "Do not consider: Golden (reference) answer, Retrieved knowledge-base chunks."
)

KNOWLEDGE_BASE_JUDGE_PROMPT: str = (
    'Adherence to Knowledge Base (score key "knowledge_base"):\n'
    "Judge ONLY whether the answer's claims are supported by the retrieved "
    "knowledge-base chunks (groundedness / hallucination detection).\n"
    "- Break the answer into its distinct factual claims. A claim is supported ONLY "
    "if the specific fact it asserts is explicitly stated in the chunk text (a "
    "verbatim or trivially reworded restatement). A claim that is merely plausible, "
    "on the same topic as a chunk, or that requires an inferential leap the chunks do "
    "not spell out is UNSUPPORTED, not supported.\n"
    "- Identify the answer's load-bearing (material) claims — the ones that carry its "
    "substance. If ANY material claim is unsupported, cap the score at 0.3 regardless "
    "of how many minor claims are supported. Otherwise score = supported claims / "
    "total factual claims.\n"
    "- Text that makes no factual claim (a greeting, a pleasantry, or a plain refusal "
    "to answer) is EXCLUDED from the claim count — do not let it inflate the score.\n"
    "- Judge groundedness ONLY, not correctness, completeness, or "
    "instruction-following. A claim faithful to the chunks is grounded even if the "
    "chunks are themselves wrong.\n"
    "- Do NOT use any outside knowledge; the retrieved chunks are the ONLY allowed "
    "source of support.\n"
    "Reasoning: quote the exact chunk span supporting the main claim. When the score "
    "is below 1.0, name the specific unsupported or invented claim.\n"
    "When scoring THIS metric, consider only these input blocks: Generated answer, "
    "Retrieved knowledge-base chunks.\n"
    "Do not consider: Assistant's configured instructions, Question, Golden "
    "(reference) answer."
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
    # Verdict band; present only on numeric judge-metric scores (v2 runs), never
    # on cosine or unscoreable/"N/A" entries.
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
