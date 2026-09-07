"""v2 judge stage: resolve the run's inputs and grade every judgeable row.

Wraps the single-row judge in `judge.py` with the run-level concerns — which
rows can be judged, the shared config prompt, the worker pool, and the run-level
summary each metric rolls up to.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from openai import OpenAI
from pydantic import ValidationError
from sqlmodel import Session

from app.core.config import settings
from app.crud.evaluations.core import resolve_evaluation_config
from app.crud.evaluations.judge import (
    JudgeInputEnum,
    JudgeMetricSpec,
    JudgeResult,
    build_judge_params,
    judge_row,
)
from app.crud.evaluations.score import SummaryScore
from app.models.evaluation import EvaluationRun
from app.models.llm.request import TextLLMParams

logger = logging.getLogger(__name__)

# Judge tells the template apart from the instructions above it.
PROMPT_TEMPLATE_LABEL = "Prompt template wrapped around each user input:"

_CHUNK_SEPARATOR = "\n---\n"


def resolve_config_prompt(
    *, session: Session, eval_run: EvaluationRun, log_prefix: str
) -> str | None:
    """The evaluated bot's own configured prompt, or None if unresolvable.

    The prompt template is appended when the config carries one, since it is
    equally part of what the bot was told to do. Returns None when the config
    carries no instructions, so the caller drops the prompt metric rather than
    grading against "".
    """
    if not eval_run.config_id or not eval_run.config_version:
        return None

    config, error = resolve_evaluation_config(
        session=session,
        config_id=eval_run.config_id,
        config_version=eval_run.config_version,
        project_id=eval_run.project_id,
    )
    if error or config is None:
        return None

    # Native/proxy params aren't TextLLMParams-shaped; a mismatch just means there
    # are no instructions to grade against, not a run failure.
    try:
        params = TextLLMParams.model_validate(config.completion.params)
    except ValidationError as exc:
        logger.info(
            f"[resolve_config_prompt] {log_prefix} Completion params are not text "
            f"params; prompt metric unscoreable | error={exc}"
        )
        return None

    sections: list[str] = []
    if params.instructions:
        sections.append(params.instructions.strip())
    if config.prompt_template and config.prompt_template.template:
        sections.append(
            f"{PROMPT_TEMPLATE_LABEL}\n{config.prompt_template.template.strip()}"
        )
    return "\n\n".join(sections) if sections else None


def build_judge_inputs(
    *, response: dict[str, Any], config_prompt: str
) -> dict[JudgeInputEnum, str]:
    """One row's judge input blocks; an empty value drops the metrics needing it."""
    return {
        JudgeInputEnum.CONFIG_PROMPT: config_prompt,
        JudgeInputEnum.QUESTION: response.get("question", ""),
        JudgeInputEnum.GENERATED_ANSWER: response.get("generated_output", ""),
        JudgeInputEnum.GOLDEN_ANSWER: response.get("ground_truth", ""),
        JudgeInputEnum.RETRIEVED_CHUNKS: _CHUNK_SEPARATOR.join(
            chunk.get("text", "")
            for chunk in (response.get("retrieved_chunks") or [])
            if chunk.get("text")
        ),
    }


def select_judgeable_rows(
    response_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rows with both a generated and a golden answer; empty sides are unscoreable."""
    return [
        response
        for response in response_results
        if response.get("generated_output") and response.get("ground_truth")
    ]


def judge_rows(
    *,
    session: Session,
    openai_client: OpenAI,
    metrics: list[JudgeMetricSpec],
    config_prompt: str,
    judgeable: list[dict[str, Any]],
    item_refs: dict[str, str],
    log_prefix: str,
) -> tuple[dict[str, JudgeResult], set[str], str | None]:
    """Run one combined judge completion per judgeable row, isolated per row.

    Returns the per-item results, the refs whose judging failed, and the judge
    model. `metrics` is the full registry; `judge_row` drops the ones a given row
    cannot supply inputs for. `config_prompt` is "" when the run's config carried
    no instructions, which drops the prompt metric for every row.
    """
    results: dict[str, JudgeResult] = {}
    failed_refs: set[str] = set()
    if not judgeable:
        return results, failed_refs, None

    # Built once per run: judging is system-config only, so every metric uses its
    # built-in prompt + shared model. Instructions vary per row (by applicable-metric
    # subset) and are composed inside judge_row.
    try:
        base_params = build_judge_params(session=session)
    except Exception as exc:
        logger.error(
            f"[judge_rows] {log_prefix} Judge setup failed; leaving all rows "
            f"unjudged | error={exc}",
            exc_info=True,
        )
        return results, {item_refs[r["item_id"]] for r in judgeable}, None

    max_workers = max(1, min(settings.EVAL_JUDGE_CONCURRENCY, len(judgeable)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                judge_row,
                openai_client=openai_client,
                base_params=base_params,
                metrics=metrics,
                inputs=build_judge_inputs(
                    response=response, config_prompt=config_prompt
                ),
            ): response["item_id"]
            for response in judgeable
        }
        for future in as_completed(future_map):
            item_id = future_map[future]
            try:
                results[item_id] = future.result()
            except Exception as exc:
                ref = item_refs[item_id]
                failed_refs.add(ref)
                logger.warning(
                    f"[judge_rows] {log_prefix} Judge failed for row; flagged "
                    f"unscoreable | item_id={item_id} | ref={ref} | error={exc}"
                )

    return results, failed_refs, base_params.get("model")


def build_metric_summary_scores(
    *, metrics: list[JudgeMetricSpec], judge_results: dict[str, JudgeResult]
) -> list[SummaryScore]:
    """Run-level summary score per metric that graded at least one row.

    Per-row scores and reasoning live on the traces, which is what the read path
    serves; only the aggregate belongs on the run.
    """
    summary_scores: list[SummaryScore] = []
    for spec in metrics:
        values = [
            metric_score.score
            for result in judge_results.values()
            if (metric_score := result.metrics.get(spec.key)) is not None
        ]
        if not values:
            continue
        array = np.array(values)
        summary_scores.append(
            {
                "name": spec.score_name,
                "avg": round(float(np.mean(array)), 2),
                "std": round(float(np.std(array)), 2),
                "total_pairs": len(values),
                "data_type": "NUMERIC",
            }
        )
    return summary_scores
