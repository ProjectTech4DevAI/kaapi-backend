"""Kickoff + scoring for the eval-iterate-improve LangGraph loop.

The loop itself (`StateGraph`, nodes, checkpoint/resume) lives in
`iteration_graph.py`. This module is the request-side entry point
(`validate_and_start_evaluation_iteration`, called from the route) and the one
scoring helper (`compute_round_scores`) shared by `wait_eval_node`.
"""

import logging
from statistics import mean
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from app.celery.utils import start_evaluation_iteration_round
from app.core.config import settings
from app.crud.evaluations.iteration import (
    create_evaluation_iteration_run,
    update_evaluation_iteration_run,
)
from app.crud.evaluations.score import (
    GROUND_TRUTH_SCORE_NAME,
    KNOWLEDGE_BASE_SCORE_NAME,
    PROMPT_SCORE_NAME,
)
from app.models.evaluation import EvaluationRun
from app.models.evaluation_iteration import (
    EvaluationIterationRun,
    EvaluationIterationRunUpdate,
    EvaluationIterationStatusEnum,
)
from app.services.evaluations.fast import validate_fast_evaluation_inputs

logger = logging.getLogger(__name__)

# `EvaluationIterationRun.stop_reason` / `EvaluationIterationReportPublic.stop_reason`
# values, set by the graph's `wait_eval_node` / `wait_improve_node`.
STOP_REASON_CEILING_REACHED = "ceiling_reached"
STOP_REASON_MAX_ROUNDS_REACHED = "max_rounds_reached"
STOP_REASON_ROUND_FAILED = "round_failed"


def validate_and_start_evaluation_iteration(
    *,
    session: Session,
    dataset_id: int,
    experiment_name: str,
    config_id: UUID,
    config_version: int,
    max_rounds: int | None,
    callback_url: str,
    organization_id: int,
    project_id: int,
    trace_id: str = "N/A",
) -> EvaluationIterationRun:
    """Validate preconditions, create the thin tracking row, and dispatch round 1.

    Reuses `validate_fast_evaluation_inputs` (dataset/config checks) unchanged —
    always `is_judge_run=True` since the loop only ever runs judged (v2) evals.
    `max_rounds` is clamped to `EVAL_ITERATION_MAX_ROUNDS_HARD_CAP`, not rejected,
    so a caller passing an oversized value degrades to the safety cap instead of
    failing outright.
    """
    resolved_max_rounds = min(
        max_rounds or settings.EVAL_ITERATION_MAX_ROUNDS_DEFAULT,
        settings.EVAL_ITERATION_MAX_ROUNDS_HARD_CAP,
    )

    logger.info(
        f"[validate_and_start_evaluation_iteration] Starting iteration loop | "
        f"dataset_id={dataset_id} | experiment_name={experiment_name} | "
        f"max_rounds={resolved_max_rounds} | org_id={organization_id} | "
        f"project_id={project_id}"
    )

    validate_fast_evaluation_inputs(
        session=session,
        dataset_id=dataset_id,
        config_id=config_id,
        config_version=config_version,
        organization_id=organization_id,
        project_id=project_id,
        is_judge_run=True,
    )

    iteration_run = create_evaluation_iteration_run(
        session=session,
        dataset_id=dataset_id,
        experiment_name=experiment_name,
        config_id=config_id,
        initial_config_version=config_version,
        callback_url=callback_url,
        organization_id=organization_id,
        project_id=project_id,
    )

    try:
        task_id = start_evaluation_iteration_round(
            iteration_run_id=iteration_run.id,
            resume=False,
            organization_id=organization_id,
            project_id=project_id,
            max_rounds=resolved_max_rounds,
            config_version=config_version,
            trace_id=trace_id,
        )
    except Exception as exc:
        logger.error(
            f"[validate_and_start_evaluation_iteration] Failed to enqueue | "
            f"iteration_run_id={iteration_run.id} | error={exc}",
            exc_info=True,
        )
        update_evaluation_iteration_run(
            session=session,
            iteration_run=iteration_run,
            update=EvaluationIterationRunUpdate(
                status=EvaluationIterationStatusEnum.FAILED,
                error_message=f"Failed to queue evaluation iteration: {exc}",
            ),
        )
        raise HTTPException(
            status_code=500,
            detail="evaluation_iteration_enqueue_failed: could not queue the loop",
        )

    logger.info(
        f"[validate_and_start_evaluation_iteration] Enqueued | "
        f"iteration_run_id={iteration_run.id} | task_id={task_id}"
    )
    return iteration_run


def compute_round_scores(eval_run: EvaluationRun) -> tuple[float, float | None] | None:
    """Derive a round's (stop_score, kb_score) from a completed judged run.

    stop_score = mean(Adherence to Ground Truth, Adherence to Prompt) — the only
    metrics that gate the stop condition. kb_score (Adherence to Knowledge Base)
    is returned for visibility only; `None` when absent since it never gates.

    Returns `None` when either required metric is missing from summary_scores —
    the caller (`wait_eval_node`) treats that the same as a failed round.
    """
    summary_scores = (eval_run.score or {}).get("summary_scores", [])
    scores_by_name = {s["name"]: s for s in summary_scores}

    ground_truth = scores_by_name.get(GROUND_TRUTH_SCORE_NAME)
    prompt = scores_by_name.get(PROMPT_SCORE_NAME)
    if ground_truth is None or prompt is None:
        logger.warning(
            f"[compute_round_scores] Missing required metric | "
            f"eval_run_id={eval_run.id} | "
            f"has_ground_truth={ground_truth is not None} | "
            f"has_prompt={prompt is not None}"
        )
        return None

    stop_score = mean([ground_truth["avg"], prompt["avg"]])
    knowledge_base = scores_by_name.get(KNOWLEDGE_BASE_SCORE_NAME)
    kb_score = knowledge_base["avg"] if knowledge_base is not None else None
    return stop_score, kb_score
