"""Human-readable AI summary of a v2 judge run's overall quality.
"""

import logging
from typing import Any

import openai
from openai import OpenAI
from sqlmodel import Session

from app.core.config import settings
from app.crud.evaluations.judge import JudgeMetricEnum
from app.crud.evaluations.response_parsing import extract_response_text
from app.crud.evaluations.score import OverallSummary
from app.services.llm.mappers import map_kaapi_to_openai_params

logger = logging.getLogger(__name__)

# Reasoning tokens count against this cap, so it needs headroom beyond the visible
# note — too low and reasoning consumes it all, yielding empty output. Length is
# bounded by the "1 to 3 sentences" instruction, not by a tight token cap.
_SUMMARY_MAX_OUTPUT_TOKENS: int = 600

# Bands for turning a metric's std (spread of its per-row scores, 0-1) into a plain
# consistency read. With repeated questions a low spread means the assistant answered
# the same question the same way each time.
_CONSISTENCY_STABLE_AT_OR_BELOW: float = 0.1
_CONSISTENCY_MIXED_AT_OR_BELOW: float = 0.2

# Internal metric key -> plain behaviour phrase, so the brief never leaks the
# "Adherence to X" labels into the model's context.
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
    'times and answers stayed consistent."'
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
    session: Session,
    openai_client: OpenAI,
    model: str,
    overall: OverallSummary,
    run_name: str,
    summary_scores: list[dict[str, Any]],
    duplication_factor: int,
) -> str | None:
    """Best-effort one-shot natural-language note on the run's overall quality.

    Reuses the aggregate's existing OpenAI client (never builds a new one) and the
    judge's reasoning-model invocation: `map_kaapi_to_openai_params` suppresses
    temperature and sets the reasoning effort, then the body goes to the Responses
    API. `summary_scores` supplies each metric's std (consistency) and
    `duplication_factor` the repeat count that the brief translates into plain
    language. A summary is a nicety, so ANY failure — missing key, API error, empty
    output — logs a warning and returns None rather than propagating: the
    deterministic overall (score/verdict/breakdown) must still persist.
    """
    user_message = _format_overall_for_prompt(
        overall=overall,
        run_name=run_name,
        summary_scores=summary_scores,
        duplication_factor=duplication_factor,
    )
    try:
        base_params, mapper_warnings = map_kaapi_to_openai_params(
            session=session,
            kaapi_params={
                "model": model,
                "effort": settings.EVAL_JUDGE_REASONING_EFFORT,
            },
        )
        if mapper_warnings:
            logger.warning(
                f"[generate_run_ai_summary] Mapper warnings: {mapper_warnings}"
            )
        params = {
            **base_params,
            "instructions": _SUMMARY_SYSTEM_PROMPT,
            "input": user_message,
            "max_output_tokens": _SUMMARY_MAX_OUTPUT_TOKENS,
        }
        response = openai_client.responses.create(**params)
    except openai.OpenAIError as exc:
        status = getattr(exc, "status_code", None)
        # 5xx is provider-side (alert-worthy); 4xx/None is caller/Kaapi-side noise.
        log = logger.error if (status and status >= 500) else logger.warning
        tag = "[OPENAI]" if status else "[KAAPI]"
        request_id = getattr(exc, "request_id", None)
        log(
            f"[generate_run_ai_summary] {tag} Summary completion failed "
            f"(code: {status or type(exc).__name__}) | model={model} | "
            f"run_name={run_name} | request_id={request_id} | {exc}",
            exc_info=True,
        )
        return None
    # Deliberately broad: a summary failure (mapper/config error, unexpected shape)
    # must never fail the run, so it degrades to a None result.
    except Exception as exc:
        logger.warning(
            f"[generate_run_ai_summary] Summary call failed; leaving ai_summary "
            f"empty | model={model} | run_name={run_name} | error={exc}",
            exc_info=True,
        )
        return None

    summary = extract_response_text(response).strip()
    if not summary:
        logger.warning(
            f"[generate_run_ai_summary] Empty summary returned | model={model} | "
            f"run_name={run_name}"
        )
        return None

    logger.info(
        f"[generate_run_ai_summary] Generated run summary | model={model} | "
        f"run_name={run_name} | chars={len(summary)}"
    )
    return summary
