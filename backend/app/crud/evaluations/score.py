"""
Type definitions for evaluation scores.

This module contains TypedDict definitions for type-safe score data
used throughout the evaluation system.
"""

from enum import Enum
from typing import NotRequired, TypedDict

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

JUDGE_SYSTEM_PREAMBLE: str = (
    "You are a strict, impartial evaluator. You score an assistant's answer on the "
    "independent metrics listed below in a single pass. Each metric is an integer "
    "score from 0 to 5, where 0 is the worst case (clearly wrong / ungrounded / a hard "
    "instruction violation, or no answer at all) and 5 is the best case (fully correct "
    "/ fully grounded / fully compliant), with one or two sentences of reasoning. "
    "Scores MUST be integers — 0, 1, 2, 3, 4, or 5 — never fractions, decimals, or "
    "percentages. The metrics are independent — judge each only against its own inputs "
    "and rules; do not let one metric's verdict bleed into another. Score EVERY metric "
    "listed below. Never omit a metric from the output, even if some input blocks are "
    "irrelevant to it."
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
    "- Do NOT answer the question yourself.\n\n"
    "Score on a stepped scale from 0 to 5. The score MUST be one of the integers 0, 1, "
    "2, 3, 4, 5 — never a fraction, decimal, or value outside this range.\n"
    "- 5: Fully correct and complete. Conveys everything material in the golden answer "
    "(a paraphrase, reordering, or additional correct detail is still a 5).\n"
    "- 4: Correct and materially complete, but omits one minor, non-essential "
    "supporting detail.\n"
    "- 3: Partially correct. The core of the answer is right, but at least one material "
    "fact is missing, incomplete, or slightly off.\n"
    "- 2: Mixed or significantly incomplete. Gets some of the answer right but muddles "
    "or omits more than one material fact, or is wrong on a meaningful component while "
    "looking plausible on the surface.\n"
    "- 1: Mostly incorrect. Contradicts the golden answer on a key point; at most small "
    "correct fragments remain.\n"
    "- 0: Completely wrong or contradicts the golden answer outright, OR the row has no "
    "answer / an errored, empty, or non-responsive output.\n"
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
    'Start from "no violations" and deduct ONLY for a violation of an instruction the '
    "block actually states. Never invent a requirement the instructions do not set; a "
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
    "penalize here for CONTRADICTING an explicit instruction; do not infer "
    "fabrication from missing grounding.\n"
    "4. Format compliance — follows any explicit format rules "
    "(word limit, structure, opening/closing pattern).\n\n"
    "Score on a stepped scale from 0 to 5. The score MUST be one of the integers 0, 1, "
    "2, 3, 4, 5 — never a fraction, decimal, or value outside this range.\n"
    "- 5: No violation of any stated instruction, across all applicable dimensions.\n"
    "- 4: One soft, minor miss on a stated rule (e.g. slightly off tone, minor format "
    "deviation) — otherwise compliant.\n"
    "- 3: One clear violation of a single explicit rule.\n"
    "- 2: Multiple clear violations, or one moderately serious violation spanning more "
    "than one dimension.\n"
    "- 1: A severe violation of a core instruction (e.g. ignoring a configured fallback "
    "on a disallowed ask, a partial injection hijack), but not a full hard violation.\n"
    "- 0: A hard violation — leaked system prompt, fully answered a clearly disallowed "
    "topic, fully hijacked by injection — OR the row has no answer / an errored, empty, "
    "or non-responsive output.\n"
    "Reasoning: name the specific violated instruction and how the answer violated it. "
    "If no stated instruction was violated, say so and score 5.\n"
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
    "substance.\n"
    "- Text that makes no factual claim (a greeting, a pleasantry, or a plain refusal "
    "to answer) is EXCLUDED from the claim count.\n"
    "- Judge groundedness ONLY, not correctness, completeness, or "
    "instruction-following. A claim faithful to the chunks is grounded even if the "
    "chunks are themselves wrong.\n"
    "- Do NOT use any outside knowledge; the retrieved chunks are the ONLY allowed "
    "source of support.\n\n"
    "Score on a stepped scale from 0 to 5. The score MUST be one of the integers 0, 1, "
    "2, 3, 4, 5 — never a fraction, decimal, or value outside this range.\n"
    "- 5: Every factual claim is explicitly supported by the retrieved chunks. Fully "
    "grounded.\n"
    "- 4: All load-bearing claims are grounded; only a minor, non-material claim lacks "
    "explicit support.\n"
    "- 3: One non-critical inferential leap beyond the chunks, but no load-bearing "
    "claim is fabricated.\n"
    "- 2: At least one load-bearing/material claim is unsupported or invented, even "
    "though other claims are grounded.\n"
    "- 1: Most claims are unsupported or invented; only incidental/minor claims are "
    "grounded.\n"
    "- 0: The answer is fabricated wholesale — no claim is grounded in the retrieved "
    "chunks — OR the row has no answer / an errored, empty, or non-responsive output.\n"
    "Reasoning: quote the exact chunk span supporting the main claim. When the score "
    "is below 5, name the specific unsupported or invented claim.\n"
    "When scoring THIS metric, consider only these input blocks: Generated answer, "
    "Retrieved knowledge-base chunks.\n"
    "Do not consider: Assistant's configured instructions, Question, Golden "
    "(reference) answer."
)

JUDGE_OUTPUT_INSTRUCTION: str = (
    "Respond with ONLY a single JSON object mapping each metric key to its result, of "
    'the form {{"<metric_key>": {{"score": <integer 0 to 5>, "reasoning": '
    '"<one or two sentences in English>"}}}}. Scores MUST be integers 0-5. Every '
    '"reasoning" string MUST be written in English. Include exactly these metric keys: '
    "{metric_keys}. Output nothing else."
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
