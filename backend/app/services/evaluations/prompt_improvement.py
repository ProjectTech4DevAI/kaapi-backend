"""AI-assisted prompt improvement service.

Analyses a completed evaluation run's weak signals (consistently low-scoring
questions and underperforming categories) and uses Claude to draft an improved
prompt, persisting the result as a new config_version.
"""

import json
import logging
from typing import Any

import anthropic
from anthropic import Anthropic
from fastapi import HTTPException
from sqlmodel import Session

from app.core.config import settings
from app.crud.config.config import get_config_by_id
from app.crud.config.version import ConfigVersionCrud
from app.crud.evaluations.core import get_evaluation_run_by_id
from app.models.config.config import ConfigTag
from app.models.config.version import (
    ConfigVersion,
    ConfigVersionPublic,
    ConfigVersionUpdate,
)
from app.models.evaluation import EvaluationRun

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

# Maximum tokens the LLM should produce for the improvement response.
_LLM_MAX_TOKENS = 4096

# JSON keys expected in the LLM's structured response.
_LLM_KEY_INSTRUCTIONS = "improved_instructions"
_LLM_KEY_RATIONALE = "rationale"

COMMIT_MESSAGE_MAX_LENGTH = 512

# Prefix that marks a commit_message as AI-generated; used as a search token for
# audit queries and by the test suite to assert provenance.
AI_GENERATED_MARKER = "[AI Generated]"


# ── internal data shapes ──────────────────────────────────────────────────────


class _LowScoringQuestion:
    """A question group that consistently scores below threshold across repetitions."""

    __slots__ = (
        "question",
        "llm_answer",
        "ground_truth_answer",
        "category",
        "mean_score",
    )

    def __init__(
        self,
        *,
        question: str,
        llm_answer: str,
        ground_truth_answer: str,
        category: str,
        mean_score: float,
    ) -> None:
        self.question = question
        self.llm_answer = llm_answer
        self.ground_truth_answer = ground_truth_answer
        self.category = category
        self.mean_score = mean_score


class _LowScoringCategory:
    """A question category whose mean metric score falls below threshold."""

    __slots__ = ("category", "avg_score")

    def __init__(self, *, category: str, avg_score: float) -> None:
        self.category = category
        self.avg_score = avg_score


# ── public entry point ────────────────────────────────────────────────────────


def improve_prompt(
    *,
    session: Session,
    evaluation_id: int,
    organization_id: int,
    project_id: int,
    metric: str,
    threshold: float,
) -> ConfigVersionPublic:
    """Run the full prompt-improvement flow synchronously and return the new version.

    Raises HTTPException for all domain errors so the route stays thin.
    """
    logger.info(
        f"[improve_prompt] Starting | evaluation_id={evaluation_id} "
        f"metric={metric} threshold={threshold} project_id={project_id}"
    )

    run = _load_completed_run(
        session=session,
        evaluation_id=evaluation_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    source_version = _resolve_source_version(
        session=session,
        run=run,
        project_id=project_id,
    )

    _verify_metric(run=run, metric=metric)

    weak_questions, questions_truncated = _select_weak_questions(
        run=run,
        metric=metric,
        threshold=threshold,
    )

    weak_categories, categories_truncated = _select_weak_categories(
        run=run,
        metric=metric,
        threshold=threshold,
    )

    truncated = questions_truncated or categories_truncated

    if not weak_questions and not weak_categories:
        raise HTTPException(
            status_code=422,
            detail="no_weak_signals: no consistently-low questions and no underperforming categories found",
        )

    logger.info(
        f"[improve_prompt] Weak signals | evaluation_id={evaluation_id} "
        f"weak_questions={len(weak_questions)} weak_categories={len(weak_categories)} "
        f"truncated={truncated}"
    )

    current_instructions = _extract_instructions(source_version)
    improved_instructions, rationale = _draft_improved_prompt(
        current_instructions=current_instructions,
        weak_questions=weak_questions,
        weak_categories=weak_categories,
        metric=metric,
        threshold=threshold,
    )

    # Provenance is embedded in commit_message (no dedicated columns) so the
    # evaluation run id, metric, and threshold remain auditable without a schema change.
    raw_commit_message = (
        f"{AI_GENERATED_MARKER} {rationale} "
        f"(source_evaluation_run_id={evaluation_id}, "
        f"metric={metric}, threshold={threshold})"
    )
    commit_message = raw_commit_message[:COMMIT_MESSAGE_MAX_LENGTH]

    version_crud = ConfigVersionCrud(
        session=session,
        config_id=run.config_id,
        project_id=project_id,
    )
    new_version = version_crud.create_or_raise(
        ConfigVersionUpdate(
            config_blob={
                "completion": {"params": {"instructions": improved_instructions}}
            },
            commit_message=commit_message,
        )
    )

    logger.info(
        f"[improve_prompt] Done | evaluation_id={evaluation_id} "
        f"new_version_id={new_version.id} version={new_version.version}"
    )

    return ConfigVersionPublic.model_validate(new_version)


# ── helpers: validation & data loading ───────────────────────────────────────


def _load_completed_run(
    *,
    session: Session,
    evaluation_id: int,
    organization_id: int,
    project_id: int,
) -> EvaluationRun:
    run = get_evaluation_run_by_id(
        session=session,
        evaluation_id=evaluation_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="evaluation_not_found: no evaluation run with this id in the caller's project",
        )

    if run.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"evaluation_not_completed: run status is '{run.status}', must be 'completed'",
        )

    return run


def _resolve_source_version(
    *,
    session: Session,
    run: EvaluationRun,
    project_id: int,
) -> ConfigVersion:
    """Load the config + config_version referenced by the run; guard soft-deletes and tenant scope."""
    # The run's config must belong to the caller's project (config has no org_id).
    config = get_config_by_id(
        session=session,
        config_id=run.config_id,
        project_id=project_id,
    )

    if config is None:
        raise HTTPException(
            status_code=409,
            detail="source_config_unavailable: the run's config is missing, soft-deleted, or outside the caller's project",
        )

    version_crud = ConfigVersionCrud(
        session=session,
        config_id=run.config_id,
        project_id=project_id,
        tag=ConfigTag.DEFAULT,
    )
    version = version_crud.read_one(version_number=run.config_version)

    if version is None:
        raise HTTPException(
            status_code=409,
            detail="source_config_unavailable: the run's config_version is missing or soft-deleted",
        )

    return version


def _resolve_summary_score(
    run: EvaluationRun,
    metric: str,
) -> dict[str, Any]:
    """Find and return the summary_scores entry whose name matches `metric` (case-insensitive).

    Raises 422 metric_not_available when no match is found.
    """
    score = run.score or {}
    summary_scores: list[dict[str, Any]] = score.get("summary_scores") or []

    needle = metric.strip().lower()
    for entry in summary_scores:
        if (entry.get("name") or "").strip().lower() == needle:
            return entry

    raise HTTPException(
        status_code=422,
        detail=f"metric_not_available: no summary score named '{metric}' is recorded in this run",
    )


def _verify_metric(
    *,
    run: EvaluationRun,
    metric: str,
) -> None:
    """Raise 422 when the chosen metric is absent in this run."""
    _resolve_summary_score(run, metric)


# ── helpers: weak signal selection ───────────────────────────────────────────


def _score_name_matches(score_entry: dict[str, Any], metric: str) -> bool:
    """Return True when a trace-level score entry's name matches `metric` (case-insensitive)."""
    return (score_entry.get("name") or "").strip().lower() == metric.strip().lower()


def _get_trace_metric_value(
    trace: dict[str, Any],
    metric: str,
) -> float | None:
    """Extract the numeric metric value for one trace from its inline scores list.

    Returns None if the score is absent, unscoreable, or not a number.
    """
    for s in trace.get("scores") or []:
        if _score_name_matches(s, metric):
            if s.get("unscoreable"):
                return None
            val = s.get("value")
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                # Categorical values can't be cast; treat as unscoreable for this metric.
                return None
    return None


def _select_weak_questions(
    *,
    run: EvaluationRun,
    metric: str,
    threshold: float,
) -> tuple[list[_LowScoringQuestion], bool]:
    """Return (low_scoring_questions, truncated).

    Groups traces by question_id; a group qualifies when the fraction of
    repetitions scoring below threshold is >= MIN_CONSISTENCY_RATIO.
    Groups with no scoreable repetitions for the chosen metric are skipped.
    Result is sorted by mean score ascending (worst first), then capped.
    """
    score = run.score or {}
    traces: list[dict[str, Any]] = score.get("traces") or []

    # question_id-less traces each form a singleton group keyed by trace_id
    # because they have no question-level identity to aggregate on.
    groups: dict[Any, list[dict[str, Any]]] = {}
    for trace in traces:
        group_key = trace.get("question_id") or trace.get("trace_id")
        groups.setdefault(group_key, []).append(trace)

    candidates: list[tuple[float, _LowScoringQuestion]] = []

    for group in groups.values():
        scoreable = [
            (t, v)
            for t in group
            if (v := _get_trace_metric_value(t, metric)) is not None
        ]
        if not scoreable:
            continue

        below = [v for _, v in scoreable if v < threshold]
        ratio = len(below) / len(scoreable)

        if ratio < settings.PROMPT_IMPROVEMENT_MIN_CONSISTENCY_RATIO:
            continue

        # All repetitions share the same question text and ground_truth;
        # use the first trace to represent the group.
        first_trace = group[0]
        mean_score = sum(v for _, v in scoreable) / len(scoreable)
        candidates.append(
            (
                mean_score,
                _LowScoringQuestion(
                    question=first_trace.get("question") or "",
                    llm_answer=first_trace.get("llm_answer") or "",
                    ground_truth_answer=first_trace.get("ground_truth_answer") or "",
                    category=first_trace.get("category") or "",
                    mean_score=mean_score,
                ),
            )
        )

    candidates.sort(key=lambda t: t[0])

    cap = settings.PROMPT_IMPROVEMENT_MAX_WEAK_QUESTIONS
    truncated = len(candidates) > cap
    return [item for _, item in candidates[:cap]], truncated


def _select_weak_categories(
    *,
    run: EvaluationRun,
    metric: str,
    threshold: float,
) -> tuple[list[_LowScoringCategory], bool]:
    """Return (low_scoring_categories, truncated).

    Computed generically from traces so it works for any score name — not from
    category_metrics, which only carries avg_cosine/avg_correctness for the two
    built-in scores and won't exist for arbitrary Langfuse scorer names.

    Groups scoreable traces by category; keeps categories whose mean score is below
    threshold; sorts ascending; truncates to cap.
    """
    score = run.score or {}
    traces: list[dict[str, Any]] = score.get("traces") or []

    category_values: dict[str, list[float]] = {}
    for trace in traces:
        val = _get_trace_metric_value(trace, metric)
        if val is None:
            continue
        category = (trace.get("category") or "").strip() or "Other"
        category_values.setdefault(category, []).append(val)

    candidates: list[_LowScoringCategory] = []
    for category, values in category_values.items():
        avg = sum(values) / len(values)
        if avg < threshold:
            candidates.append(_LowScoringCategory(category=category, avg_score=avg))

    candidates.sort(key=lambda cat: cat.avg_score)

    cap = settings.PROMPT_IMPROVEMENT_MAX_WEAK_CATEGORIES
    truncated = len(candidates) > cap
    return candidates[:cap], truncated


# ── helpers: LLM call ─────────────────────────────────────────────────────────


def _build_improvement_prompt(
    *,
    current_instructions: str,
    weak_questions: list[_LowScoringQuestion],
    weak_categories: list[_LowScoringCategory],
    metric: str,
    threshold: float,
) -> str:
    """Build the user message text sent to Claude."""
    question_lines = "\n".join(
        f"  {i+1}. Question: {item.question}\n"
        f"     LLM answer: {item.llm_answer}\n"
        f"     Ground truth: {item.ground_truth_answer}\n"
        f"     Category: {item.category} | Mean {metric}: {item.mean_score:.3f}"
        for i, item in enumerate(weak_questions)
    )

    category_lines = "\n".join(
        f"  - {cat.category} (avg {metric}: {cat.avg_score:.3f})"
        for cat in weak_categories
    )

    return (
        f"You are a prompt engineer. Your task is to improve the following system prompt "
        f"so that an AI assistant performs better on the weak areas identified by an evaluation.\n\n"
        f"## Current system prompt\n```\n{current_instructions}\n```\n\n"
        f"## Evaluation metric\n{metric} (threshold: {threshold})\n\n"
        f"## Questions the assistant answered poorly (consistently below threshold)\n"
        f"{question_lines if question_lines else '  (none)'}\n\n"
        f"## Underperforming question categories (average score below threshold)\n"
        f"{category_lines if category_lines else '  (none)'}\n\n"
        "## Instructions\n"
        "1. Rewrite the system prompt to address the weaknesses above.\n"
        "2. Do NOT change the model, knowledge base, or any other configuration — only the prompt text.\n"
        "3. Keep every part of the prompt that is working well.\n"
        "4. Respond ONLY with a JSON object (no markdown fences) with exactly two keys:\n"
        '   - "improved_instructions": the full rewritten prompt text (string)\n'
        '   - "rationale": one short paragraph explaining what you targeted and why (string)\n'
    )


def _draft_improved_prompt(
    *,
    current_instructions: str,
    weak_questions: list[_LowScoringQuestion],
    weak_categories: list[_LowScoringCategory],
    metric: str,
    threshold: float,
) -> tuple[str, str]:
    """Call Claude to produce improved instructions + rationale.

    Returns (improved_instructions, rationale).
    Raises HTTPException(502) on any LLM failure or unusable output.
    """
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: the platform Anthropic key "
                "(ANTHROPIC_API_KEY) is not configured"
            ),
        )

    user_message = _build_improvement_prompt(
        current_instructions=current_instructions,
        weak_questions=weak_questions,
        weak_categories=weak_categories,
        metric=metric,
        threshold=threshold,
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=settings.PROMPT_IMPROVEMENT_MODEL,
            max_tokens=_LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        logger.info(
            f"[_draft_improved_prompt] LLM call succeeded | "
            f"model={settings.PROMPT_IMPROVEMENT_MODEL} response_id={response.id}"
        )

    except anthropic.AuthenticationError as e:
        logger.warning(
            f"[_draft_improved_prompt] [ANTHROPIC] Authentication failed "
            f"(code: 401): Verify the ANTHROPIC_API_KEY is "
            f"valid, not expired, and configured correctly.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: Anthropic authentication failed — "
                "verify the platform API key is valid and not expired"
            ),
        )

    except anthropic.RateLimitError as e:
        logger.warning(
            f"[_draft_improved_prompt] [ANTHROPIC] Rate limit exceeded "
            f"(code: 429): Hit Anthropic rate/quota — wait ≥1 min and retry.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: Anthropic rate limit exceeded — "
                "wait at least 1 minute and retry"
            ),
        )

    except anthropic.APITimeoutError as e:
        # Must come before APIConnectionError — APITimeoutError is a subclass.
        logger.error(
            f"[_draft_improved_prompt] [KAAPI] Anthropic request timed out "
            f"(code: {type(e).__name__}): retry with a smaller payload.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: Anthropic request timed out — "
                "retry. If persistent, contact Kaapi"
            ),
        )

    except anthropic.APIConnectionError as e:
        logger.error(
            f"[_draft_improved_prompt] [KAAPI] Anthropic connection failed "
            f"(code: {type(e).__name__}): network or DNS issue reaching Anthropic.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: network error reaching Anthropic — "
                "check connectivity. If persistent, contact Kaapi"
            ),
        )

    except anthropic.APIStatusError as e:
        status = e.status_code
        # 5xx is provider-side (alert-worthy); 4xx is caller's fault (noise if alerted)
        log = logger.error if status and status >= 500 else logger.warning
        log(
            f"[_draft_improved_prompt] [ANTHROPIC] API status error "
            f"(code: {status}): {e.message}.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"prompt_generation_failed: Anthropic returned HTTP {status} — "
                "retry or contact Kaapi if persistent"
            ),
        )

    except Exception as e:
        logger.error(
            f"[_draft_improved_prompt] [KAAPI] Unexpected error during LLM call "
            f"(code: {type(e).__name__}): not raised by the Anthropic SDK — "
            f"likely a Kaapi-side failure. Contact Kaapi if persistent.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: unexpected error during prompt generation — "
                "contact Kaapi if persistent"
            ),
        )

    return _parse_llm_response(raw_text)


def _parse_llm_response(raw_text: str) -> tuple[str, str]:
    """Extract improved_instructions and rationale from the LLM JSON response.

    Raises HTTPException(502) when the response is not parseable or missing keys.
    """
    # Strip optional markdown code fences the model sometimes emits despite instructions.
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        stripped = "\n".join(inner).strip()

    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            f"[_parse_llm_response] Could not parse LLM JSON response | error={exc}"
        )
        raise HTTPException(
            status_code=502,
            detail="prompt_generation_failed: LLM returned a response that could not be parsed as JSON",
        )

    instructions = data.get(_LLM_KEY_INSTRUCTIONS)
    rationale = data.get(_LLM_KEY_RATIONALE)

    if not instructions or not isinstance(instructions, str):
        raise HTTPException(
            status_code=502,
            detail="prompt_generation_failed: LLM response missing 'improved_instructions' field",
        )
    if not rationale or not isinstance(rationale, str):
        raise HTTPException(
            status_code=502,
            detail="prompt_generation_failed: LLM response missing 'rationale' field",
        )

    return instructions.strip(), rationale.strip()


# ── helpers: blob manipulation ────────────────────────────────────────────────


def _extract_instructions(version: ConfigVersion) -> str:
    """Read completion.params.instructions from the source config_blob."""
    blob: dict[str, Any] = version.config_blob or {}
    return blob.get("completion", {}).get("params", {}).get("instructions") or ""
