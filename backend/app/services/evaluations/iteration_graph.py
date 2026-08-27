"""LangGraph orchestration for the eval-iterate-improve loop.

`StateGraph` cycle: start_eval -> wait_eval -> (conditional) -> {finalize, start_improve}
start_improve -> wait_improve -> start_eval (loop-back).

Cyclic-graph + checkpoint/resume are the only reason LangGraph is used here — the
loop's own stop/continue decisions are deterministic (delta-threshold, round cap),
not LLM-planned. The only LLM call anywhere in the loop is the existing
prompt-drafting call inside `execute_prompt_improvement`, invoked unchanged via
`start_prompt_improvement_job`.

Rule for every node: open and close its own short-lived DB session inside the node
function. A pause at `interrupt()` can span many cron ticks (hours), so nothing
DB-related may survive between calls except what's persisted in the checkpoint
(LangGraph-owned) or the thin `EvaluationIterationRun` row.
"""

import logging
from functools import lru_cache
from typing import Any, TypedDict
from uuid import UUID

from celery.exceptions import SoftTimeLimitExceeded
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sqlmodel import Session

from app.core.config import settings
from app.core.db import engine
from app.crud.evaluations.core import TERMINAL_EVAL_STATUSES, get_evaluation_run_by_id
from app.crud.evaluations.iteration import (
    get_evaluation_iteration_run_by_id,
    update_evaluation_iteration_run,
)
from app.crud.jobs import JobCrud
from app.models.evaluation_iteration import (
    EvaluationIterationReportPublic,
    EvaluationIterationRoundPublic,
    EvaluationIterationRunUpdate,
    EvaluationIterationStatusEnum,
)
from app.models.job import JobStatus
from app.services.evaluations.fast import validate_and_start_fast_evaluation
from app.services.evaluations.iteration import (
    STOP_REASON_CEILING_REACHED,
    STOP_REASON_MAX_ROUNDS_REACHED,
    STOP_REASON_ROUND_FAILED,
    compute_round_scores,
)
from app.services.evaluations.prompt_improvement import start_prompt_improvement_job
from app.utils import APIResponse, get_webhook_secret, send_callback

logger = logging.getLogger(__name__)

_JOB_WAITING_STATUSES = {JobStatus.PENDING, JobStatus.PROCESSING}


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


def _psycopg_conn_string() -> str:
    """Derive a plain psycopg conninfo string from the app's SQLAlchemy DSN.

    langgraph-checkpoint-postgres connects via psycopg (v3) directly rather than
    through the SQLAlchemy engine, but the app's DSN already targets the psycopg
    driver (`postgresql+psycopg://`) — stripping the SQLAlchemy dialect qualifier
    is the only adaptation needed.
    """
    return str(settings.SQLALCHEMY_DATABASE_URI).replace(
        "postgresql+psycopg://", "postgresql://", 1
    )


@lru_cache(maxsize=1)
def get_evaluation_iteration_checkpointer() -> PostgresSaver:
    """Module-level singleton checkpointer, backed by its own small connection pool.

    `.setup()` creates the checkpoint tables (`checkpoints`, `checkpoint_blobs`,
    `checkpoint_writes`) — schema owned by the library, not Alembic. It's a
    `CREATE TABLE IF NOT EXISTS`-style call, so it's safe to run on every first
    access rather than gating it behind a separate startup hook.
    """
    pool = ConnectionPool(
        conninfo=_psycopg_conn_string(),
        min_size=1,
        max_size=5,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    logger.info("[get_evaluation_iteration_checkpointer] Checkpointer ready")
    return checkpointer


def start_eval_node(state: EvaluationIterationState) -> dict[str, Any]:
    """Kick off this round's judged fast-eval run."""
    run_name = (
        f"{state['experiment_name']}-iter{state['iteration_run_id']}-"
        f"r{state['round_number']}"
    )
    with Session(engine) as session:
        eval_run = validate_and_start_fast_evaluation(
            session=session,
            dataset_id=state["dataset_id"],
            run_name=run_name,
            config_id=UUID(state["config_id"]),
            config_version=state["config_version"],
            organization_id=state["organization_id"],
            project_id=state["project_id"],
            is_judge_run=True,
        )
    logger.info(
        f"[start_eval_node] Round eval started | "
        f"iteration_run_id={state['iteration_run_id']} | "
        f"round_number={state['round_number']} | eval_run_id={eval_run.id}"
    )
    return {"current_eval_run_id": eval_run.id}


def wait_eval_node(state: EvaluationIterationState) -> dict[str, Any]:
    """Poll the round's eval run; score + decide continue/stop once it's terminal.

    A `Command(resume=True)` replays this node from the top, and `interrupt()` only
    re-pauses for a call position that has no queued resume yet — so a single
    interrupt-then-return falls straight through on every resume after the first.
    Looping the fetch+check+interrupt forces a fresh (unconsumed) interrupt() call
    each tick the run is still processing, so it correctly re-pauses instead of
    proceeding as if the run were done.
    """
    current_eval_run_id = state["current_eval_run_id"]
    if current_eval_run_id is None:
        # Invariant: start_eval_node always sets this before wait_eval_node runs.
        return {
            "stop_reason": STOP_REASON_ROUND_FAILED,
            "error_message": "wait_eval_node reached with no current_eval_run_id set",
        }

    with Session(engine) as session:
        while True:
            eval_run = get_evaluation_run_by_id(
                session=session,
                evaluation_id=current_eval_run_id,
                organization_id=state["organization_id"],
                project_id=state["project_id"],
            )
            if eval_run is None:
                return {
                    "stop_reason": STOP_REASON_ROUND_FAILED,
                    "error_message": f"EvaluationRun {current_eval_run_id} not found",
                }
            if eval_run.status in TERMINAL_EVAL_STATUSES:
                break
            interrupt({"waiting_on": "eval", "eval_run_id": eval_run.id})

        if eval_run.status == "failed":
            return {
                "stop_reason": STOP_REASON_ROUND_FAILED,
                "error_message": eval_run.error_message or "Evaluation run failed",
            }

        scores = compute_round_scores(eval_run)
        if scores is None:
            return {
                "stop_reason": STOP_REASON_ROUND_FAILED,
                "error_message": (
                    "Missing required judge metric (Adherence to Ground Truth / "
                    "Adherence to Prompt) in eval_run.score.summary_scores"
                ),
            }
        eval_run_id = eval_run.id

    stop_score, kb_score = scores
    round_entry = {
        "round_number": state["round_number"],
        "eval_run_id": eval_run_id,
        "config_version": state["config_version"],
        "stop_score": stop_score,
        "kb_score": kb_score,
    }
    history = [*state["history"], round_entry]

    best_stop_score = state.get("best_stop_score")
    best_round_number = state.get("best_round_number")
    best_config_version = state.get("best_config_version")
    if best_stop_score is None or stop_score > best_stop_score:
        best_stop_score = stop_score
        best_round_number = state["round_number"]
        best_config_version = state["config_version"]

    previous_scores = [entry["stop_score"] for entry in state["history"]]
    consecutive_low_delta_rounds = 0
    if previous_scores:
        delta = stop_score - previous_scores[-1]
        if delta < settings.EVAL_ITERATION_CEILING_DELTA_THRESHOLD:
            consecutive_low_delta_rounds = (
                state.get("consecutive_low_delta_rounds", 0) + 1
            )

    update: dict[str, Any] = {
        "history": history,
        "best_stop_score": best_stop_score,
        "best_round_number": best_round_number,
        "best_config_version": best_config_version,
        "consecutive_low_delta_rounds": consecutive_low_delta_rounds,
    }
    if (
        consecutive_low_delta_rounds
        >= settings.EVAL_ITERATION_CEILING_CONSECUTIVE_ROUNDS
    ):
        update["stop_reason"] = STOP_REASON_CEILING_REACHED
    elif state["round_number"] >= state["max_rounds"]:
        update["stop_reason"] = STOP_REASON_MAX_ROUNDS_REACHED

    logger.info(
        f"[wait_eval_node] Round scored | iteration_run_id={state['iteration_run_id']} | "
        f"round_number={state['round_number']} | stop_score={stop_score} | "
        f"consecutive_low_delta_rounds={consecutive_low_delta_rounds} | "
        f"stop_reason={update.get('stop_reason')}"
    )
    return update


def route_after_eval(state: EvaluationIterationState) -> str:
    return "finalize_node" if state.get("stop_reason") else "start_improve_node"


def start_improve_node(state: EvaluationIterationState) -> dict[str, Any]:
    """Draft the next prompt version from this round's judge traces.

    `callback_url=""` is a deliberate no-op — `_send_improve_prompt_callback`
    already skips the HTTP round-trip on an empty URL, since only the loop's own
    `finalize_node` callback is user-facing.
    """
    current_eval_run_id = state["current_eval_run_id"]
    if current_eval_run_id is None:
        # Invariant: only reached via route_after_eval, after wait_eval_node set this.
        raise ValueError("start_improve_node reached with no current_eval_run_id set")

    with Session(engine) as session:
        job = start_prompt_improvement_job(
            session=session,
            evaluation_id=current_eval_run_id,
            organization_id=state["organization_id"],
            project_id=state["project_id"],
            callback_url="",
            require_judge_run=True,
        )
    logger.info(
        f"[start_improve_node] Prompt improvement job started | "
        f"iteration_run_id={state['iteration_run_id']} | job_id={job.id}"
    )
    return {"current_improvement_job_id": str(job.id)}


def wait_improve_node(state: EvaluationIterationState) -> dict[str, Any]:
    """Poll the prompt-improvement job; advance the round once it's terminal.

    Loops the fetch+check+interrupt (see `wait_eval_node` docstring for why): a
    resume replays the node from the top, and a single interrupt-then-return would
    fall through on every resume after the first instead of re-pausing.
    """
    current_improvement_job_id = state["current_improvement_job_id"]
    if current_improvement_job_id is None:
        # Invariant: start_improve_node always sets this before wait_improve_node runs.
        raise ValueError(
            "wait_improve_node reached with no current_improvement_job_id set"
        )

    with Session(engine) as session:
        while True:
            job = JobCrud(session=session).get(
                job_id=UUID(current_improvement_job_id),
                project_id=state["project_id"],
            )
            if job is None:
                return {
                    "stop_reason": STOP_REASON_ROUND_FAILED,
                    "error_message": (
                        f"Prompt improvement job {current_improvement_job_id} not found"
                    ),
                }
            if job.status not in _JOB_WAITING_STATUSES:
                break
            interrupt({"waiting_on": "improve", "job_id": str(job.id)})

        if job.status == JobStatus.FAILED:
            return {
                "stop_reason": STOP_REASON_ROUND_FAILED,
                "error_message": job.error_message or "Prompt improvement job failed",
            }

        new_version = (job.meta or {}).get("version")

    if new_version is None:
        return {
            "stop_reason": STOP_REASON_ROUND_FAILED,
            "error_message": "Prompt improvement job succeeded without a version in meta",
        }

    logger.info(
        f"[wait_improve_node] Prompt improved | "
        f"iteration_run_id={state['iteration_run_id']} | "
        f"next_round_number={state['round_number'] + 1} | config_version={new_version}"
    )
    return {
        "round_number": state["round_number"] + 1,
        "config_version": new_version,
        "current_improvement_job_id": None,
    }


def _build_iteration_report(
    state: EvaluationIterationState, status: EvaluationIterationStatusEnum
) -> EvaluationIterationReportPublic:
    history = [EvaluationIterationRoundPublic(**entry) for entry in state["history"]]
    best_round = next(
        (r for r in history if r.round_number == state.get("best_round_number")),
        None,
    )
    return EvaluationIterationReportPublic(
        iteration_run_id=state["iteration_run_id"],
        status=status,
        stop_reason=state.get("stop_reason"),
        best_round=best_round,
        history=history,
        error_message=state.get("error_message"),
    )


def finalize_node(state: EvaluationIterationState) -> dict[str, Any]:
    """Terminal node: persist the thin row and POST the report to callback_url."""
    stop_reason = state.get("stop_reason")
    if stop_reason is None:
        # Invariant: route_after_eval only reaches this node once stop_reason is set.
        logger.warning(
            f"[finalize_node] Reached with no stop_reason set | "
            f"iteration_run_id={state['iteration_run_id']}"
        )
        stop_reason = STOP_REASON_MAX_ROUNDS_REACHED
    status = (
        EvaluationIterationStatusEnum.FAILED
        if stop_reason == STOP_REASON_ROUND_FAILED
        else EvaluationIterationStatusEnum.COMPLETED
    )

    with Session(engine) as session:
        iteration_run = get_evaluation_iteration_run_by_id(
            session=session,
            iteration_run_id=state["iteration_run_id"],
            organization_id=state["organization_id"],
            project_id=state["project_id"],
        )
        if iteration_run is None:
            logger.error(
                f"[finalize_node] EvaluationIterationRun not found | "
                f"iteration_run_id={state['iteration_run_id']}"
            )
            return {}

        update_evaluation_iteration_run(
            session=session,
            iteration_run=iteration_run,
            update=EvaluationIterationRunUpdate(
                status=status,
                stop_reason=stop_reason,
                error_message=state.get("error_message"),
            ),
        )

    report = _build_iteration_report(state, status)
    error_message = state.get("error_message")
    envelope = (
        APIResponse.failure_response(
            error=error_message, data=report.model_dump(mode="json")
        )
        if error_message
        else APIResponse.success_response(data=report.model_dump(mode="json"))
    )
    webhook_secret = get_webhook_secret(state["project_id"], state["organization_id"])
    send_callback(
        state["callback_url"], envelope.model_dump(), webhook_secret=webhook_secret
    )

    logger.info(
        f"[finalize_node] Loop finished | iteration_run_id={state['iteration_run_id']} | "
        f"status={status.value} | stop_reason={stop_reason} | "
        f"rounds={len(state['history'])}"
    )
    return {}


def build_evaluation_iteration_graph(checkpointer: PostgresSaver) -> CompiledStateGraph:
    graph = StateGraph(EvaluationIterationState)
    graph.add_node("start_eval_node", start_eval_node)
    graph.add_node("wait_eval_node", wait_eval_node)
    graph.add_node("start_improve_node", start_improve_node)
    graph.add_node("wait_improve_node", wait_improve_node)
    graph.add_node("finalize_node", finalize_node)

    graph.add_edge(START, "start_eval_node")
    graph.add_edge("start_eval_node", "wait_eval_node")
    graph.add_conditional_edges(
        "wait_eval_node",
        route_after_eval,
        {"finalize_node": "finalize_node", "start_improve_node": "start_improve_node"},
    )
    graph.add_edge("start_improve_node", "wait_improve_node")
    graph.add_conditional_edges(
        "wait_improve_node",
        route_after_eval,
        {"finalize_node": "finalize_node", "start_improve_node": "start_eval_node"},
    )
    graph.add_edge("finalize_node", END)

    return graph.compile(checkpointer=checkpointer)


def _build_initial_state(
    *,
    iteration_run_id: int,
    organization_id: int,
    project_id: int,
    max_rounds: int | None,
    config_version: int | None,
) -> EvaluationIterationState:
    with Session(engine) as session:
        iteration_run = get_evaluation_iteration_run_by_id(
            session=session,
            iteration_run_id=iteration_run_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if iteration_run is None:
            raise ValueError(f"EvaluationIterationRun {iteration_run_id} not found")

        return EvaluationIterationState(
            iteration_run_id=iteration_run.id,
            dataset_id=iteration_run.dataset_id,
            experiment_name=iteration_run.experiment_name,
            config_id=str(iteration_run.config_id),
            config_version=config_version or iteration_run.initial_config_version,
            round_number=1,
            max_rounds=max_rounds or settings.EVAL_ITERATION_MAX_ROUNDS_DEFAULT,
            current_eval_run_id=None,
            current_improvement_job_id=None,
            history=[],
            best_round_number=None,
            best_config_version=None,
            best_stop_score=None,
            consecutive_low_delta_rounds=0,
            stop_reason=None,
            error_message=None,
            organization_id=iteration_run.organization_id,
            project_id=iteration_run.project_id,
            callback_url=iteration_run.callback_url,
        )


def _mark_iteration_run_failed(
    *, iteration_run_id: int, organization_id: int, project_id: int, error_message: str
) -> None:
    """Fail a loop from a fresh session so a killed task leaves no dangling row."""
    try:
        with Session(engine) as session:
            iteration_run = get_evaluation_iteration_run_by_id(
                session=session,
                iteration_run_id=iteration_run_id,
                organization_id=organization_id,
                project_id=project_id,
            )
            if (
                iteration_run is None
                or iteration_run.status != EvaluationIterationStatusEnum.PROCESSING
            ):
                return
            update_evaluation_iteration_run(
                session=session,
                iteration_run=iteration_run,
                update=EvaluationIterationRunUpdate(
                    status=EvaluationIterationStatusEnum.FAILED,
                    error_message=error_message,
                ),
            )
            callback_url = iteration_run.callback_url

        envelope = APIResponse.failure_response(error=error_message)
        webhook_secret = get_webhook_secret(project_id, organization_id)
        send_callback(
            callback_url, envelope.model_dump(), webhook_secret=webhook_secret
        )

        logger.info(
            f"[_mark_iteration_run_failed] iteration_run_id={iteration_run_id} marked failed"
        )
    except Exception:
        logger.error(
            f"[_mark_iteration_run_failed] Could not mark iteration_run_id="
            f"{iteration_run_id} failed",
            exc_info=True,
        )


def _run_graph_step(
    *,
    iteration_run_id: int,
    resume: bool,
    organization_id: int,
    project_id: int,
    max_rounds: int | None,
    config_version: int | None,
) -> None:
    checkpointer = get_evaluation_iteration_checkpointer()
    graph = build_evaluation_iteration_graph(checkpointer)
    thread_config = {"configurable": {"thread_id": str(iteration_run_id)}}

    if resume:
        if checkpointer.get_tuple(thread_config) is None:
            logger.warning(
                f"[_run_graph_step] No checkpoint found for resume | "
                f"iteration_run_id={iteration_run_id}"
            )
            return
        graph.invoke(Command(resume=True), config=thread_config)
        return

    initial_state = _build_initial_state(
        iteration_run_id=iteration_run_id,
        organization_id=organization_id,
        project_id=project_id,
        max_rounds=max_rounds,
        config_version=config_version,
    )
    graph.invoke(initial_state, config=thread_config)


def execute_evaluation_iteration_graph_step(
    *,
    iteration_run_id: int,
    resume: bool,
    organization_id: int,
    project_id: int,
    max_rounds: int | None = None,
    config_version: int | None = None,
) -> None:
    """Guarded entrypoint: advance one graph step, never leave the thin row dangling.

    Either re-interrupts (checkpoint already persisted by LangGraph, thin row stays
    PROCESSING) or reaches `finalize_node` (which already updated the thin row and
    sent the callback before this returns).
    """
    logger.info(
        f"[execute_evaluation_iteration_graph_step] Starting | "
        f"iteration_run_id={iteration_run_id} | resume={resume}"
    )
    try:
        _run_graph_step(
            iteration_run_id=iteration_run_id,
            resume=resume,
            organization_id=organization_id,
            project_id=project_id,
            max_rounds=max_rounds,
            config_version=config_version,
        )
    except SoftTimeLimitExceeded:
        logger.error(
            f"[execute_evaluation_iteration_graph_step] Soft time limit | "
            f"iteration_run_id={iteration_run_id}"
        )
        _mark_iteration_run_failed(
            iteration_run_id=iteration_run_id,
            organization_id=organization_id,
            project_id=project_id,
            error_message="Evaluation iteration step exceeded the time limit.",
        )
        raise
    except Exception:
        logger.error(
            f"[execute_evaluation_iteration_graph_step] Unexpected failure | "
            f"iteration_run_id={iteration_run_id}",
            exc_info=True,
        )
        _mark_iteration_run_failed(
            iteration_run_id=iteration_run_id,
            organization_id=organization_id,
            project_id=project_id,
            error_message="Evaluation iteration loop failed unexpectedly.",
        )
        raise
