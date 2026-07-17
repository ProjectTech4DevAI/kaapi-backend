"""
Runs one OpenAI judge call per row to score all enabled metrics together.
Executes after cosine similarity, so judge failures never block cosine; metrics are registry-driven and easily extensible.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import openai
from openai import OpenAI
from sqlmodel import Session
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.core.config import settings
from app.crud.evaluations.score import (
    GROUND_TRUTH_JUDGE_PROMPT,
    GROUND_TRUTH_SCORE_NAME,
    JUDGE_OUTPUT_INSTRUCTION,
    JUDGE_SYSTEM_PREAMBLE,
)
from app.services.llm.mappers import map_kaapi_to_openai_params

logger = logging.getLogger(__name__)


class JudgeMetricEnum(str, Enum):
    """Registry keys — also the JSON keys the combined judge returns per metric."""

    GROUND_TRUTH = "ground_truth"


class JudgeInputEnum(str, Enum):
    """Per-row inputs a metric may require in the composed judge prompt."""

    QUESTION = "question"
    GENERATED_ANSWER = "generated_answer"
    GOLDEN_ANSWER = "golden_answer"


_INPUT_LABELS: dict[JudgeInputEnum, str] = {
    JudgeInputEnum.QUESTION: "Question",
    JudgeInputEnum.GENERATED_ANSWER: "Generated answer",
    JudgeInputEnum.GOLDEN_ANSWER: "Golden (reference) answer",
}


@dataclass(frozen=True)
class JudgeMetricSpec:
    """Everything the pipeline needs to run and persist one judge metric."""

    key: JudgeMetricEnum
    score_name: str
    prompt_fragment: str
    required_inputs: tuple[JudgeInputEnum, ...]
    per_item_column: str
    cost_stage: str


# Phase 1: only ground_truth, knowledge_base / prompt slot in here. All
# metrics are graded by one combined call, so they share a single judge model
# (settings.EVAL_JUDGE_MODEL); there is no per-metric model.
METRIC_REGISTRY: dict[JudgeMetricEnum, JudgeMetricSpec] = {
    JudgeMetricEnum.GROUND_TRUTH: JudgeMetricSpec(
        key=JudgeMetricEnum.GROUND_TRUTH,
        score_name=GROUND_TRUTH_SCORE_NAME,
        prompt_fragment=GROUND_TRUTH_JUDGE_PROMPT,
        required_inputs=(
            JudgeInputEnum.QUESTION,
            JudgeInputEnum.GENERATED_ANSWER,
            JudgeInputEnum.GOLDEN_ANSWER,
        ),
        per_item_column="per_item_ground_truth",
        cost_stage="ground_truth_judge",
    ),
}


def enabled_metric_specs() -> list[JudgeMetricSpec]:
    """The metrics a judged run scores. Phase 1 always scores every registry entry."""
    return list(METRIC_REGISTRY.values())


# Per-call retry mechanism (mirrors the fast-eval Responses/Embeddings stages).
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_SECONDS = 1.0
_RETRY_MAX_DELAY_SECONDS = 30.0

_RETRYABLE_OPENAI_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

# reraise=True so the call-site handler sees the original OpenAIError.
_retry_judge_call = retry(
    retry=retry_if_exception_type(_RETRYABLE_OPENAI_ERRORS),
    wait=wait_random_exponential(
        multiplier=_RETRY_BASE_DELAY_SECONDS, max=_RETRY_MAX_DELAY_SECONDS
    ),
    stop=stop_after_attempt(_RETRY_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,
)


@dataclass
class MetricScore:
    """One metric's outcome for a row: score in [0, 1] and its reasoning."""

    score: float
    reasoning: str


@dataclass
class JudgeResult:
    """One row's combined judge outcome across all enabled metrics, plus usage."""

    metrics: dict[JudgeMetricEnum, MetricScore]
    usage: dict[str, int]


def build_judge_params(
    *,
    session: Session,
    metrics: list[JudgeMetricSpec],
) -> tuple[dict[str, Any], str]:
    """Build the model-independent OpenAI body and the combined judge system prompt.

    Judging is system-config only: every metric uses its built-in rubric fragment,
    and the single combined call runs on one shared judge model
    (settings.EVAL_JUDGE_MODEL) for all metrics. The system prompt is the shared
    preamble followed by each enabled metric's fragment.
    """
    judge_params: dict[str, Any] = {
        "model": settings.EVAL_JUDGE_MODEL,
        "effort": settings.EVAL_JUDGE_REASONING_EFFORT,
    }

    base_params, mapper_warnings = map_kaapi_to_openai_params(
        session=session, kaapi_params=judge_params
    )
    if mapper_warnings:
        logger.warning(f"[build_judge_params] Mapper warnings: {mapper_warnings}")

    fragments = "\n\n".join(spec.prompt_fragment for spec in metrics)
    system_prompt = f"{JUDGE_SYSTEM_PREAMBLE}\n\n{fragments}"
    # The judge prompt IS the instructions; overwrite anything the mapper carried.
    base_params["instructions"] = system_prompt
    return base_params, system_prompt


def _compose_judge_input(
    *, metrics: list[JudgeMetricSpec], inputs: dict[JudgeInputEnum, str]
) -> str:
    """Append the row's inputs (union of enabled metrics' needs) + the output contract.

    Metric prompts carry no interpolation placeholder — inputs are appended here.
    """
    required: list[JudgeInputEnum] = []
    for spec in metrics:
        for key in spec.required_inputs:
            if key not in required:
                required.append(key)

    # Render in stable enum order regardless of registry declaration order.
    blocks = [
        f"{_INPUT_LABELS[key]}:\n{inputs.get(key, '')}"
        for key in JudgeInputEnum
        if key in required
    ]
    metric_keys = ", ".join(spec.key.value for spec in metrics)
    contract = JUDGE_OUTPUT_INSTRUCTION.format(metric_keys=metric_keys)
    return "\n\n".join(blocks) + "\n\n" + contract


def _extract_response_text(response: Any) -> str:
    """Extract generated text, preferring `output_text` then walking `output`."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    output = getattr(response, "output", None) or []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", None)
                if text:
                    return text
    return ""


def _parse_metric_score(key: JudgeMetricEnum, raw: Any) -> MetricScore:
    """Parse one metric's {"score", "reasoning"} object. Raises ValueError if malformed."""
    if not isinstance(raw, dict):
        raise ValueError(f"metric '{key.value}' is not a JSON object: {raw!r}")
    if "score" not in raw:
        raise ValueError(f"metric '{key.value}' missing 'score': {raw}")
    try:
        score = float(raw["score"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metric '{key.value}' score is not a number: {raw.get('score')!r}"
        ) from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"metric '{key.value}' score out of [0, 1]: {score}")

    reasoning = str(raw.get("reasoning") or "").strip()
    if not reasoning:
        raise ValueError(f"metric '{key.value}' has empty 'reasoning'")
    return MetricScore(score=score, reasoning=reasoning)


def _parse_judge_output(
    text: str, metrics: list[JudgeMetricSpec]
) -> dict[JudgeMetricEnum, MetricScore]:
    """Parse the combined reply into a per-metric score map.

    Leniently extracts the outermost JSON object so a stray prose wrapper doesn't
    break parsing. A well-formed object missing one metric leaves only that metric
    unscoreable (dropped from the map); a fully malformed reply raises so the caller
    isolates the whole row. Raises ValueError on anything unparseable.
    """
    if not text or not text.strip():
        raise ValueError("empty judge response")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in judge response: {text[:200]!r}")

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in judge response: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"judge response is not a JSON object: {data!r}")

    results: dict[JudgeMetricEnum, MetricScore] = {}
    for spec in metrics:
        raw = data.get(spec.key.value)
        if raw is None:
            # Well-formed but missing this metric: only this metric is unscoreable.
            continue
        results[spec.key] = _parse_metric_score(spec.key, raw)

    if not results:
        raise ValueError(f"judge response scored no enabled metric: {data}")
    return results


@_retry_judge_call
def _create_judge_response(openai_client: OpenAI, params: dict[str, Any]) -> Any:
    return openai_client.responses.create(**params)


def judge_row(
    *,
    openai_client: OpenAI,
    base_params: dict[str, Any],
    metrics: list[JudgeMetricSpec],
    question: str,
    generated_answer: str,
    golden_answer: str,
) -> JudgeResult:
    """Run one combined judge completion for a row and return its per-metric result.

    `base_params` is the model-independent body from `build_judge_params`, built
    once per run; only `input` varies per row. Raises on retry-exhausted OpenAI
    errors or malformed output so the caller can flag the whole row unscoreable.
    """
    model = base_params.get("model")
    params = {
        **base_params,
        "input": _compose_judge_input(
            metrics=metrics,
            inputs={
                JudgeInputEnum.QUESTION: question,
                JudgeInputEnum.GENERATED_ANSWER: generated_answer,
                JudgeInputEnum.GOLDEN_ANSWER: golden_answer,
            },
        ),
    }

    try:
        response = _create_judge_response(openai_client, params)
    except openai.OpenAIError as exc:
        status = getattr(exc, "status_code", None)
        # 5xx is provider-side (alert-worthy); 4xx/None is caller/Kaapi-side noise.
        log = logger.error if (status and status >= 500) else logger.warning
        tag = "[OPENAI]" if status else "[KAAPI]"
        request_id = getattr(exc, "request_id", None)
        log(
            f"[judge_row] {tag} Judge completion failed "
            f"(code: {status or type(exc).__name__}) | model={model} | "
            f"request_id={request_id} | {exc}",
            exc_info=True,
        )
        raise

    usage_obj = getattr(response, "usage", None)
    usage = {
        "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }

    metric_scores = _parse_judge_output(_extract_response_text(response), metrics)
    return JudgeResult(metrics=metric_scores, usage=usage)
