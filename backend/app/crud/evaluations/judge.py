"""Native LLM-as-a-judge for v2 fast evaluations.

One OpenAI call per row scores all enabled metrics together. v2 runs are judge-only
(no cosine, no embeddings) and v1 never invokes this judge. A per-row failure is
isolated to that row; a metric whose run-level input cannot be resolved is dropped
for the run, which still completes on the remaining metrics.
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
from app.crud.evaluations.response_parsing import (
    extract_response_text as _extract_response_text,
)
from app.crud.evaluations.score import (
    GROUND_TRUTH_JUDGE_PROMPT,
    GROUND_TRUTH_SCORE_NAME,
    JUDGE_OUTPUT_INSTRUCTION,
    JUDGE_SYSTEM_PREAMBLE,
    KNOWLEDGE_BASE_JUDGE_PROMPT,
    KNOWLEDGE_BASE_SCORE_NAME,
    PROMPT_JUDGE_PROMPT,
    PROMPT_SCORE_NAME,
)
from app.services.llm.mappers import map_kaapi_to_openai_params

logger = logging.getLogger(__name__)


class JudgeMetricEnum(str, Enum):
    """Registry keys — also the JSON keys the combined judge returns per metric."""

    GROUND_TRUTH = "ground_truth"
    PROMPT = "prompt"
    KNOWLEDGE_BASE = "knowledge_base"


class JudgeInputEnum(str, Enum):
    """Per-row inputs a metric may require in the composed judge prompt.

    Declaration order fixes the render order in the input block, so new inputs are
    appended last to keep existing metrics' prompts byte-identical.
    """

    CONFIG_PROMPT = "config_prompt"
    QUESTION = "question"
    GENERATED_ANSWER = "generated_answer"
    GOLDEN_ANSWER = "golden_answer"
    RETRIEVED_CHUNKS = "retrieved_chunks"


_INPUT_LABELS: dict[JudgeInputEnum, str] = {
    JudgeInputEnum.CONFIG_PROMPT: "Assistant's configured instructions",
    JudgeInputEnum.QUESTION: "Question",
    JudgeInputEnum.GENERATED_ANSWER: "Generated answer",
    JudgeInputEnum.GOLDEN_ANSWER: "Golden (reference) answer",
    JudgeInputEnum.RETRIEVED_CHUNKS: "Retrieved knowledge-base chunks",
}

RUN_LEVEL_INPUTS: frozenset[JudgeInputEnum] = frozenset({JudgeInputEnum.CONFIG_PROMPT})

JUDGE_COST_STAGE: str = "judge"

# Every input block is visible to every metric, so each metric section states its own
# scope — at the input level, never the metric level, so it can never be read as
# permission to skip the metric itself.
_CONSIDER_INPUTS_TEMPLATE: str = (
    "When scoring THIS metric, consider only these input blocks: {labels}."
)
_IGNORE_INPUTS_TEMPLATE: str = "Do not consider: {labels}."
_LABEL_SEPARATOR: str = ", "


@dataclass(frozen=True)
class JudgeMetricSpec:
    """Everything the pipeline needs to run and persist one judge metric."""

    key: JudgeMetricEnum
    score_name: str
    prompt_fragment: str
    required_inputs: tuple[JudgeInputEnum, ...]


# All metrics are graded by one combined call, so they share a single judge model
# (settings.EVAL_JUDGE_MODEL); there is no per-metric model. A metric only runs on a
# row that carries all its required_inputs.
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
    ),
    JudgeMetricEnum.PROMPT: JudgeMetricSpec(
        key=JudgeMetricEnum.PROMPT,
        score_name=PROMPT_SCORE_NAME,
        prompt_fragment=PROMPT_JUDGE_PROMPT,
        required_inputs=(
            JudgeInputEnum.CONFIG_PROMPT,
            JudgeInputEnum.QUESTION,
            JudgeInputEnum.GENERATED_ANSWER,
        ),
    ),
    JudgeMetricEnum.KNOWLEDGE_BASE: JudgeMetricSpec(
        key=JudgeMetricEnum.KNOWLEDGE_BASE,
        score_name=KNOWLEDGE_BASE_SCORE_NAME,
        prompt_fragment=KNOWLEDGE_BASE_JUDGE_PROMPT,
        # No QUESTION: groundedness judges the answer against the chunks alone.
        required_inputs=(
            JudgeInputEnum.GENERATED_ANSWER,
            JudgeInputEnum.RETRIEVED_CHUNKS,
        ),
    ),
}


def enabled_metric_specs(
    *, available_run_inputs: frozenset[JudgeInputEnum] = frozenset()
) -> list[JudgeMetricSpec]:
    """The metrics a run scores: those whose run-level inputs it could resolve."""
    specs: list[JudgeMetricSpec] = []
    for spec in METRIC_REGISTRY.values():
        missing = (set(spec.required_inputs) & RUN_LEVEL_INPUTS) - available_run_inputs
        if missing:
            continue
        specs.append(spec)
    return specs


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


def _required_input_union(metrics: list[JudgeMetricSpec]) -> list[JudgeInputEnum]:
    """Inputs needed by at least one enabled metric, in stable enum order.

    Single source for both the rendered input blocks and the per-metric scoping lines,
    so the stated scope can never drift from what is actually sent.
    """
    required = {key for spec in metrics for key in spec.required_inputs}
    return [key for key in JudgeInputEnum if key in required]


def _format_input_labels(keys: list[JudgeInputEnum]) -> str:
    return _LABEL_SEPARATOR.join(_INPUT_LABELS[key] for key in keys)


def _build_metric_section(*, spec: JudgeMetricSpec, union: list[JudgeInputEnum]) -> str:
    """One metric's rubric fragment plus its registry-derived input scoping."""
    considered = [key for key in union if key in spec.required_inputs]
    ignored = [key for key in union if key not in spec.required_inputs]

    lines = [
        spec.prompt_fragment,
        _CONSIDER_INPUTS_TEMPLATE.format(labels=_format_input_labels(considered)),
    ]
    # With nothing out of scope (single enabled metric), an "ignore nothing" sentence
    # would be noise the model could misread.
    if ignored:
        lines.append(
            _IGNORE_INPUTS_TEMPLATE.format(labels=_format_input_labels(ignored))
        )
    return "\n".join(lines)


def build_judge_params(*, session: Session) -> dict[str, Any]:
    """Build the model-independent OpenAI body for the combined judge call.

    Judging is system-config only: the single combined call runs on one shared
    judge model (settings.EVAL_JUDGE_MODEL) for all metrics. `instructions` are NOT
    baked here — they're the applicable-metric subset, which varies per row, so
    `judge_row` composes and sets them from `_compose_system_prompt`.
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

    return base_params


def _compose_system_prompt(metrics: list[JudgeMetricSpec]) -> str:
    """Shared preamble plus each metric's fragment and its input scoping.

    The combined call shows every input block to every metric, so each section
    states its own CONSIDER/IGNORE scope over the union of these metrics' inputs —
    keeping the stated scope aligned with what `_compose_judge_input` actually sends.
    """
    union = _required_input_union(metrics)
    sections = "\n\n".join(
        _build_metric_section(spec=spec, union=union) for spec in metrics
    )
    return f"{JUDGE_SYSTEM_PREAMBLE}\n\n{sections}"


def _applicable_metrics(
    metrics: list[JudgeMetricSpec], inputs: dict[JudgeInputEnum, str]
) -> list[JudgeMetricSpec]:
    """The metrics whose required_inputs are all present and non-empty for this row.

    A row missing an input (e.g. no retrieved chunks) simply omits that metric from
    the judge call — it stays unscoreable for that row, not scored 0.
    """
    return [
        spec
        for spec in metrics
        if all(inputs.get(key, "").strip() for key in spec.required_inputs)
    ]


def _compose_judge_input(
    *, metrics: list[JudgeMetricSpec], inputs: dict[JudgeInputEnum, str]
) -> str:
    """Append the row's inputs (union of enabled metrics' needs) + the output contract.

    Metric prompts carry no interpolation placeholder — inputs are appended here.
    """
    blocks = [
        f"{_INPUT_LABELS[key]}:\n{inputs.get(key, '')}"
        for key in _required_input_union(metrics)
    ]
    metric_keys = ", ".join(spec.key.value for spec in metrics)
    contract = JUDGE_OUTPUT_INSTRUCTION.format(metric_keys=metric_keys)
    return "\n\n".join(blocks) + "\n\n" + contract


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
    break parsing. A metric missing from an otherwise valid object is simply dropped;
    anything unparseable raises ValueError so the caller isolates the whole row.
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
    inputs: dict[JudgeInputEnum, str],
) -> JudgeResult:
    """Run one combined judge completion for a row and return its per-metric result.

    `base_params` is the model-independent body from `build_judge_params`, built
    once per run; the system prompt and input block are composed here from the
    per-row applicable-metric subset (a metric whose inputs are absent is dropped,
    so its key never enters the prompt or the expected reply). `config_prompt`
    travels only as an input block, never as the judge's own `instructions`, so the
    evaluated bot's prompt cannot steer the grader. Raises on retry-exhausted OpenAI
    errors or malformed output so the caller can flag the whole row unscoreable.
    """
    applicable = _applicable_metrics(metrics, inputs)
    if not applicable:
        raise ValueError("no judge metric applies to this row's inputs")

    model = base_params.get("model")
    params = {
        **base_params,
        # The judge prompt IS the instructions; overwrite anything the mapper carried.
        "instructions": _compose_system_prompt(applicable),
        "input": _compose_judge_input(metrics=applicable, inputs=inputs),
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

    metric_scores = _parse_judge_output(_extract_response_text(response), applicable)
    return JudgeResult(metrics=metric_scores, usage=usage)
