"""Native LLM-as-a-judge correctness scoring for fast evaluations.

One OpenAI completion per evaluated row grades the generated answer against the
ground truth and returns a 0..1 correctness score plus a short reasoning. Runs
inside the fast-eval pipeline AFTER cosine similarity, so a judge failure can
never block the cosine score. Per-row isolation is the caller's job: this module
raises on any failure/malformed output so the caller can flag that single row
unscoreable and continue.

See docs/llm-judge-integration.md for the full design.
"""

import json
import logging
from dataclasses import dataclass
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
from app.crud.evaluations.core import resolve_evaluation_config
from app.crud.evaluations.score import (
    DEFAULT_JUDGE_PROMPT,
    JUDGE_OUTPUT_INSTRUCTION,
)
from app.models.llm.request import ConfigBlob, LLMCallConfig
from app.services.llm.mappers import map_kaapi_to_openai_params

logger = logging.getLogger(__name__)


# Per-call retry policy (mirrors the fast-eval Responses/Embeddings stages).
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
class JudgeResult:
    """One row's judge outcome: correctness in [0, 1], reasoning, and token usage."""

    score: float
    reasoning: str
    usage: dict[str, int]


def resolve_judge_blob(
    *,
    session: Session,
    judge_config: LLMCallConfig | None,
    project_id: int,
) -> ConfigBlob | None:
    """Resolve a run's judge_config to a ConfigBlob, or None for the zero-config default.

    A stored reference (id + version) is resolved through the existing scoped
    config flow (tenant isolation is enforced there — a config from another
    (org, project) is never resolvable); an ad-hoc blob is used directly; absent
    config returns None so the caller falls back to the built-in prompt + model.

    Raises ValueError if a stored reference cannot be resolved (the route already
    validated resolvability, so this only fires on an unexpected worker-time gap).
    """
    if judge_config is None:
        return None

    if judge_config.blob is not None:
        return judge_config.blob

    blob, error = resolve_evaluation_config(
        session=session,
        config_id=judge_config.id,
        config_version=judge_config.version,
        project_id=project_id,
    )
    if error or blob is None:
        raise ValueError(
            f"Failed to resolve stored judge config "
            f"(id={judge_config.id}, version={judge_config.version}): {error}"
        )
    return blob


def _judge_prompt(blob: ConfigBlob | None) -> str:
    """The judge system prompt: the blob's template override, else the built-in."""
    if blob is not None and blob.prompt_template and blob.prompt_template.template:
        return blob.prompt_template.template
    return DEFAULT_JUDGE_PROMPT


def build_judge_params(
    *,
    session: Session,
    blob: ConfigBlob | None,
) -> tuple[dict[str, Any], str]:
    """Build the OpenAI base body (model + sampling) and the judge system prompt.

    The judge's model and settings come from the blob's completion params (or the
    fallback model + default temperature when absent). A bot's own instructions
    and knowledge base are stripped so they can never leak into the grader — the
    judge prompt is the only instruction.
    """
    if blob is None:
        judge_params: dict[str, Any] = {
            "model": settings.EVAL_JUDGE_FALLBACK_MODEL,
            "temperature": settings.EVAL_JUDGE_DEFAULT_TEMPERATURE,
        }
    else:
        judge_params = dict(blob.completion.params or {})
        judge_params.pop("instructions", None)
        judge_params.pop("knowledge_base_ids", None)
        if not judge_params.get("model"):
            judge_params["model"] = settings.EVAL_JUDGE_FALLBACK_MODEL

    base_params, mapper_warnings = map_kaapi_to_openai_params(
        session=session, kaapi_params=judge_params
    )
    if mapper_warnings:
        logger.info(f"[build_judge_params] Mapper warnings: {mapper_warnings}")

    # The judge prompt IS the instructions; overwrite anything the mapper carried.
    base_params["instructions"] = _judge_prompt(blob)
    return base_params, base_params["instructions"]


def _compose_judge_input(
    *, question: str, generated_answer: str, ground_truth: str
) -> str:
    """Append the row's Q/A/GT plus the output contract; templates carry no placeholder."""
    return (
        f"Question:\n{question}\n\n"
        f"Generated answer:\n{generated_answer}\n\n"
        f"Ground truth answer:\n{ground_truth}\n\n"
        f"{JUDGE_OUTPUT_INSTRUCTION}"
    )


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


def _parse_judge_output(text: str) -> tuple[float, str]:
    """Parse the model reply into (score in [0,1], non-empty reasoning).

    Leniently extracts the outermost JSON object so a stray prose wrapper doesn't
    break parsing. Raises ValueError on anything malformed so the caller isolates
    the row.
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

    if "score" not in data:
        raise ValueError(f"judge response missing 'score': {data}")
    try:
        score = float(data["score"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"judge 'score' is not a number: {data.get('score')!r}"
        ) from exc

    if not 0.0 <= score <= 1.0:
        raise ValueError(f"judge score out of [0, 1]: {score}")

    reasoning = str(data.get("reasoning") or "").strip()
    if not reasoning:
        raise ValueError("judge response has empty 'reasoning'")

    return score, reasoning


@_retry_judge_call
def _create_judge_response(openai_client: OpenAI, params: dict[str, Any]) -> Any:
    return openai_client.responses.create(**params)


def judge_row(
    *,
    openai_client: OpenAI,
    base_params: dict[str, Any],
    question: str,
    generated_answer: str,
    ground_truth: str,
) -> JudgeResult:
    """Run one judge completion for a row and return its correctness result.

    `base_params` is the model-independent body from `build_judge_params`, built
    once per run; only `input` varies per row. Raises on transient-exhausted
    OpenAI errors or malformed output so the caller can flag the row unscoreable.
    """
    model = base_params.get("model")
    params = {
        **base_params,
        "input": _compose_judge_input(
            question=question,
            generated_answer=generated_answer,
            ground_truth=ground_truth,
        ),
    }

    try:
        response = _create_judge_response(openai_client, params)
    except openai.OpenAIError as exc:
        status = getattr(exc, "status_code", None)
        # 5xx is provider-side (alert-worthy); 4xx/None is caller/Kaapi-side noise.
        log = logger.error if (status and status >= 500) else logger.warning
        tag = "[OPENAI]" if status else "[KAAPI]"
        log(
            f"[judge_row] {tag} Judge completion failed "
            f"(code: {status or type(exc).__name__}) | model={model} | {exc}",
            exc_info=True,
        )
        raise

    usage_obj = getattr(response, "usage", None)
    usage = {
        "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }

    score, reasoning = _parse_judge_output(_extract_response_text(response))
    return JudgeResult(score=score, reasoning=reasoning, usage=usage)
