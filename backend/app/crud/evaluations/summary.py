"""Human-readable AI summary of a v2 judge run's overall quality.
"""

import json
import logging
from typing import Any

from app.core.config import settings
from app.crud.evaluations.judge import JudgeMetricEnum
from app.crud.evaluations.score import OverallSummary
from app.services.llm.providers.claude import ClaudeProvider, log_anthropic_error

logger = logging.getLogger(__name__)

_SUMMARY_MAX_TOKENS: int = 2000

_CONSISTENCY_STABLE_AT_OR_BELOW: float = 0.5
_CONSISTENCY_MIXED_AT_OR_BELOW: float = 1.0

_LLM_KEY_SUMMARY: str = "summary"
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {_LLM_KEY_SUMMARY: {"type": "string"}},
    "required": [_LLM_KEY_SUMMARY],
    "additionalProperties": False,
}

# Internal metric key -> plain behaviour phrase, so the brief never leaks the labels into the model's context.
_DIMENSION_PLAIN_NAME: dict[str, str] = {
    JudgeMetricEnum.GROUND_TRUTH.value: "Accuracy against the expected answers",
    JudgeMetricEnum.KNOWLEDGE_BASE.value: "Grounding in the source material",
    JudgeMetricEnum.PROMPT.value: "Tone and instruction-following",
}

_SUMMARY_SYSTEM_PROMPT: str = (
    "You write a short, warm, plain-language note about how an AI assistant did on an "
    "evaluation, the way a colleague would summarize at a glance. Write 1 to 3 "
    "sentences — no bullet points, no headings. Use no scores or numbers at all; the "
    "ONLY number you may state is how many times each question was asked. "
    "You are given an overall standing, how each area did (as a band word), and how "
    "consistent the answers were. Translate everything into plain behaviour words and "
    "NEVER use the internal area labels, raw scores, weights, deltas, or the word "
    "'verdict'. Map the areas like this: accuracy against the expected answers -> "
    "'accurate' or 'matched the expected answers'; grounding in the source material -> "
    "'grounded', 'backed by the source material', or 'not made up'; tone and "
    "instruction-following -> 'on-tone' or 'followed the instructions (language, "
    "style)'. Lead with how it did overall in that plain language, then note the "
    "strongest and any weaker area. When each question was asked more than once and the "
    "answers stayed stable, say so (e.g. 'answers stayed consistent'); if they varied, "
    "say the answers varied. Do not invent facts beyond what you are given. "
    "Style anchor — match this tone and length, do not copy the wording: "
    '"Consistently grounded and on-tone across the set. Each question was asked 5 '
    'times and answers stayed consistent." '
    'Return your answer as JSON: {"summary": "<the note>"}.'
)


def _consistency_read(std: float | None) -> str:
    """Qualitative stability of a metric's per-row scores, derived from its std."""
    if std is None:
        return "consistency unknown"
    if std <= _CONSISTENCY_STABLE_AT_OR_BELOW:
        return "answers stayed consistent"
    if std <= _CONSISTENCY_MIXED_AT_OR_BELOW:
        return "answers were mostly consistent, with some variation"
    return "answers varied"


def _format_overall_for_prompt(
    *,
    overall: OverallSummary,
    run_name: str,
    summary_scores: list[dict[str, Any]],
    duplication_factor: int,
) -> str:
    """Compact qualitative brief for the summary model — bands, not raw scores.

    Hands over the overall band, a per-area band + consistency read (from each
    metric's std), and the repetition factor. No numbers cross except the repeat
    count; the instructions do the plain-language translation.
    """
    std_by_name = {
        score["name"]: score.get("std")
        for score in summary_scores
        if isinstance(score, dict) and "name" in score
    }

    lines = [
        f"Run: {run_name}",
        f"Overall standing: {overall['verdict']}.",
        "How each area did:",
    ]
    for dim in overall["breakdown"]:
        plain_name = _DIMENSION_PLAIN_NAME.get(dim["key"], dim["name"])
        consistency = _consistency_read(std_by_name.get(dim["name"]))
        lines.append(f"- {plain_name}: {dim['verdict']}; {consistency}")

    if duplication_factor > 1:
        lines.append(f"Each question was asked {duplication_factor} times.")
    else:
        lines.append("Each question was asked once (no repetition to speak of).")

    return "\n".join(lines)


def generate_run_ai_summary(
    *,
    model: str,
    overall: OverallSummary,
    run_name: str,
    summary_scores: list[dict[str, Any]],
    duplication_factor: int,
) -> str | None:
    """Best-effort one-shot natural-language note on the run's overall quality.

    Uses the platform-owned ANTHROPIC_API_KEY (same key as prompt improvement),
    so this works without per-project Anthropic credentials.
    """
    if not settings.ANTHROPIC_API_KEY:
        logger.warning(
            "[generate_run_ai_summary] ANTHROPIC_API_KEY not configured; "
            "leaving ai_summary empty"
        )
        return None

    user_message = _format_overall_for_prompt(
        overall=overall,
        run_name=run_name,
        summary_scores=summary_scores,
        duplication_factor=duplication_factor,
    )
    client = ClaudeProvider.create_client({"api_key": settings.ANTHROPIC_API_KEY})

    try:
        response = client.messages.create(
            model=model,
            max_tokens=_SUMMARY_MAX_TOKENS,
            system=_SUMMARY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        data: dict[str, str] = json.loads(text)
        summary: str = data[_LLM_KEY_SUMMARY].strip()

    # Deliberately broad: a summary failure (typed Anthropic error, bad JSON,
    # unexpected shape) must never fail the run, so it degrades to a None
    # result regardless of cause.
    except Exception as exc:
        log_anthropic_error(
            exc,
            fn_name="generate_run_ai_summary",
            context=f"model={model} | run_name={run_name}",
        )
        return None

    if not summary:
        logger.warning(
            f"[generate_run_ai_summary] Empty summary returned | model={model} | "
            f"run_name={run_name}"
        )
        return None

    return summary
