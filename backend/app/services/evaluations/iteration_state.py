"""Graph state and round bookkeeping for the eval-iterate-improve loop.

The state lives in the LangGraph checkpoint, not on `evaluation_iteration_run`,
so a pause at `interrupt()` can span many cron ticks. Everything here is pure —
no DB, no HTTP — so the loop's stop/continue arithmetic can be read and tested
on its own.
"""

from typing import Any, TypedDict

from app.core.config import settings
from app.models.evaluation_iteration import (
    EvaluationIterationReportPublic,
    EvaluationIterationRoundPublic,
    EvaluationIterationStatusEnum,
)
from app.services.evaluations.iteration import (
    STOP_REASON_CEILING_REACHED,
    STOP_REASON_MAX_ROUNDS_REACHED,
)


class EvaluationIterationState(TypedDict):
    iteration_run_id: int
    dataset_id: int
    experiment_name: str
    config_id: str
    config_version: int
    round_number: int
    max_rounds: int
    current_eval_run_id: int | None
    current_improvement_job_id: str | None
    history: list[dict[str, Any]]
    best_round_number: int | None
    best_config_version: int | None
    best_stop_score: float | None
    consecutive_low_delta_rounds: int
    stop_reason: str | None
    error_message: str | None
    organization_id: int
    project_id: int
    callback_url: str


def _count_low_delta_rounds(
    *, state: EvaluationIterationState, stop_score: float
) -> int:
    """Consecutive rounds whose gain stayed under the ceiling threshold.

    Resets to 0 on the first round (no baseline) and on any round that clears the
    threshold.
    """
    previous_scores = [entry["stop_score"] for entry in state["history"]]
    if not previous_scores:
        return 0
    delta = stop_score - previous_scores[-1]
    if delta >= settings.EVAL_ITERATION_CEILING_DELTA_THRESHOLD:
        return 0
    return state.get("consecutive_low_delta_rounds", 0) + 1


def advance_round_state(
    *,
    state: EvaluationIterationState,
    eval_run_id: int,
    stop_score: float,
    kb_score: float | None,
) -> dict[str, Any]:
    """Record this round's result and decide whether the loop should stop.

    Returns the state delta only. `kb_score` is recorded for visibility and never
    gates stopping.
    """
    round_entry = {
        "round_number": state["round_number"],
        "eval_run_id": eval_run_id,
        "config_version": state["config_version"],
        "stop_score": stop_score,
        "kb_score": kb_score,
    }

    best_stop_score = state.get("best_stop_score")
    is_best = best_stop_score is None or stop_score > best_stop_score
    consecutive_low_delta_rounds = _count_low_delta_rounds(
        state=state, stop_score=stop_score
    )

    update: dict[str, Any] = {
        "history": [*state["history"], round_entry],
        "best_stop_score": stop_score if is_best else best_stop_score,
        "best_round_number": (
            state["round_number"] if is_best else state.get("best_round_number")
        ),
        "best_config_version": (
            state["config_version"] if is_best else state.get("best_config_version")
        ),
        "consecutive_low_delta_rounds": consecutive_low_delta_rounds,
    }

    if (
        consecutive_low_delta_rounds
        >= settings.EVAL_ITERATION_CEILING_CONSECUTIVE_ROUNDS
    ):
        update["stop_reason"] = STOP_REASON_CEILING_REACHED
    elif state["round_number"] >= state["max_rounds"]:
        update["stop_reason"] = STOP_REASON_MAX_ROUNDS_REACHED

    return update


def build_iteration_report(
    state: EvaluationIterationState, status: EvaluationIterationStatusEnum
) -> EvaluationIterationReportPublic:
    """The round-by-round report delivered to the caller's callback_url."""
    history = [EvaluationIterationRoundPublic(**entry) for entry in state["history"]]
    best_round = next(
        (r for r in history if r.round_number == state.get("best_round_number")), None
    )
    return EvaluationIterationReportPublic(
        iteration_run_id=state["iteration_run_id"],
        status=status,
        stop_reason=state.get("stop_reason"),
        best_round=best_round,
        history=history,
        error_message=state.get("error_message"),
    )
