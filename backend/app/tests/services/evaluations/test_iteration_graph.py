"""Tests for the eval-iterate-improve LangGraph loop (`iteration_graph.py`).

Every node opens its own `Session(engine)` (see the module docstring on why),
so tests redirect that to the transactional `db` fixture the same way
`test_evaluation_fast.py` does for `execute_fast_evaluation_chunk`. The one
external HTTP boundary is `send_callback` (finalize_node's webhook delivery);
`get_webhook_secret` is a DB lookup and is left real per the implementer's
mocking guidance.

Interrupt/resume is exercised through the *compiled* graph with an injected
`InMemorySaver` (per the implementer's guidance) rather than by calling
`wait_eval_node`/`wait_improve_node` directly — `interrupt()` requires a live
LangGraph runnable context (it reads a contextvar-backed config), so calling
it outside `graph.invoke(...)` raises a plain `RuntimeError`, not the
`GraphInterrupt` the real caller (the pregel executor) would swallow. The
terminal branches (failed/completed/ceiling/max_rounds) never call
`interrupt()`, so those are exercised as direct node calls.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlmodel import Session

from app.core.config import settings
from app.crud.evaluations.iteration import create_evaluation_iteration_run
from app.crud.evaluations.score import (
    GROUND_TRUTH_SCORE_NAME,
    KNOWLEDGE_BASE_SCORE_NAME,
    PROMPT_SCORE_NAME,
)
from app.crud.jobs import JobCrud
from app.models import EvaluationDataset, EvaluationRun
from app.models.evaluation import RunModeEnum
from app.models.evaluation_iteration import (
    EvaluationIterationRun,
    EvaluationIterationStatusEnum,
)
from app.models.job import JobStatus, JobType, JobUpdate
from app.services.evaluations.iteration import (
    STOP_REASON_CEILING_REACHED,
    STOP_REASON_MAX_ROUNDS_REACHED,
    STOP_REASON_ROUND_FAILED,
)
from app.services.evaluations.iteration_graph import (
    build_evaluation_iteration_graph,
    execute_evaluation_iteration_graph_step,
    finalize_node,
    route_after_eval,
    wait_eval_node,
    wait_improve_node,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)
from app.tests.utils.utils import random_lower_string

_CALLBACK_URL = "https://example.com/callback"


class _FakeSessionCtx:
    """Context manager returning the test session; `__exit__` never closes it.

    Mirrors the pattern in test_evaluation_fast.py — node code opens its own
    `Session(engine)`, which must be redirected to the test's transactional
    session or it would talk to a different, uncommitted connection.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    def __enter__(self) -> Session:
        return self._db

    def __exit__(self, *exc: object) -> bool:
        return False


def _patch_session(db: Session):
    return patch(
        "app.services.evaluations.iteration_graph.Session",
        lambda *a, **k: _FakeSessionCtx(db),
    )


def _make_dataset(db: Session, user_api_key: TestAuthContext) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


def _make_eval_run(
    db: Session,
    user_api_key: TestAuthContext,
    *,
    status: str = "processing",
    score: dict | None = None,
    error_message: str | None = None,
) -> EvaluationRun:
    dataset = _make_dataset(db, user_api_key)
    config = create_test_config(
        db=db, project_id=user_api_key.project_id, use_kaapi_schema=True
    )
    run = EvaluationRun(
        run_name=f"iter-round-{random_lower_string()}",
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        config_id=config.id,
        config_version=1,
        status=status,
        run_mode=RunModeEnum.FAST,
        is_judge_run=True,
        score=score,
        error_message=error_message,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _make_iteration_run(
    db: Session, user_api_key: TestAuthContext
) -> EvaluationIterationRun:
    dataset = _make_dataset(db, user_api_key)
    config = create_test_config(
        db=db, project_id=user_api_key.project_id, use_kaapi_schema=True
    )
    return create_evaluation_iteration_run(
        session=db,
        dataset_id=dataset.id,
        experiment_name=f"iter-{random_lower_string()}",
        config_id=config.id,
        initial_config_version=1,
        callback_url=_CALLBACK_URL,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


def _good_score(ground_truth: float, prompt: float) -> dict:
    return {
        "summary_scores": [
            {"name": GROUND_TRUTH_SCORE_NAME, "avg": ground_truth},
            {"name": PROMPT_SCORE_NAME, "avg": prompt},
        ]
    }


def _base_state(**overrides) -> dict:
    state = {
        "iteration_run_id": 1,
        "dataset_id": 1,
        "experiment_name": "exp",
        "config_id": str(uuid4()),
        "config_version": 1,
        "round_number": 1,
        "max_rounds": 10,
        "current_eval_run_id": None,
        "current_improvement_job_id": None,
        "history": [],
        "best_round_number": None,
        "best_config_version": None,
        "best_stop_score": None,
        "consecutive_low_delta_rounds": 0,
        "stop_reason": None,
        "error_message": None,
        "organization_id": 1,
        "project_id": 1,
        "callback_url": _CALLBACK_URL,
    }
    state.update(overrides)
    return state


class TestGraphInterruptAndResume:
    """One full round-trip through the compiled graph: both wait nodes pause
    on a non-terminal sub-run and resume correctly once it completes."""

    def test_eval_wait_then_improve_wait_interrupt_and_resume(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        round1_eval_run = _make_eval_run(db, user_api_key, status="processing")
        round2_eval_run = _make_eval_run(db, user_api_key, status="processing")
        eval_runs = iter([round1_eval_run, round2_eval_run])

        checkpointer = InMemorySaver()
        graph = build_evaluation_iteration_graph(checkpointer)
        thread_config = {
            "configurable": {"thread_id": f"test-thread-{round1_eval_run.id}"}
        }
        state = _base_state(
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        with (
            _patch_session(db),
            patch(
                "app.services.evaluations.iteration_graph.validate_and_start_fast_evaluation",
                side_effect=lambda **_: next(eval_runs),
            ),
        ):
            result = graph.invoke(state, config=thread_config)

        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value == {
            "waiting_on": "eval",
            "eval_run_id": round1_eval_run.id,
        }
        assert graph.get_state(thread_config).next == ("wait_eval_node",)

        round1_eval_run.status = "completed"
        round1_eval_run.score = _good_score(0.9, 0.8)
        db.add(round1_eval_run)
        db.commit()

        improve_job_holder: dict = {}

        def _fake_start_improve(**kwargs):
            job = JobCrud(session=kwargs["session"]).create(
                job_type=JobType.PROMPT_IMPROVEMENT,
                project_id=user_api_key.project_id,
            )
            improve_job_holder["job"] = job
            return job

        with (
            _patch_session(db),
            patch(
                "app.services.evaluations.iteration_graph.start_prompt_improvement_job",
                side_effect=_fake_start_improve,
            ),
        ):
            result = graph.invoke(Command(resume=True), config=thread_config)

        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value == {
            "waiting_on": "improve",
            "job_id": str(improve_job_holder["job"].id),
        }
        assert graph.get_state(thread_config).next == ("wait_improve_node",)

        JobCrud(session=db).update(
            improve_job_holder["job"].id,
            JobUpdate(status=JobStatus.SUCCESS, meta={"version": 2}),
        )

        with (
            _patch_session(db),
            patch(
                "app.services.evaluations.iteration_graph.validate_and_start_fast_evaluation",
                side_effect=lambda **_: next(eval_runs),
            ),
        ):
            result = graph.invoke(Command(resume=True), config=thread_config)

        # Looped back to start_eval_node for round 2, using the improved
        # config_version, and is now waiting on the second eval run.
        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value == {
            "waiting_on": "eval",
            "eval_run_id": round2_eval_run.id,
        }
        snapshot = graph.get_state(thread_config).values
        assert snapshot["round_number"] == 2
        assert snapshot["config_version"] == 2
        assert len(snapshot["history"]) == 1


class TestWaitEvalNodeBranches:
    def test_failed_eval_run_sets_round_failed(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        eval_run = _make_eval_run(
            db, user_api_key, status="failed", error_message="upstream boom"
        )
        state = _base_state(
            current_eval_run_id=eval_run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        with _patch_session(db):
            result = wait_eval_node(state)

        assert result["stop_reason"] == STOP_REASON_ROUND_FAILED
        assert "upstream boom" in result["error_message"]

    def test_completed_good_scores_continues_without_stop_reason(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        eval_run = _make_eval_run(
            db, user_api_key, status="completed", score=_good_score(0.9, 0.7)
        )
        state = _base_state(
            current_eval_run_id=eval_run.id,
            round_number=1,
            max_rounds=10,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        with _patch_session(db):
            result = wait_eval_node(state)

        assert "stop_reason" not in result
        assert len(result["history"]) == 1
        assert result["history"][0]["stop_score"] == pytest.approx(0.8)
        assert result["best_round_number"] == 1

    def test_third_consecutive_low_delta_round_triggers_ceiling_reached(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        # Round 4's stop_score (0.72) sits within EVAL_ITERATION_CEILING_DELTA_THRESHOLD
        # of round 3's (0.715) — the third consecutive low-delta round.
        eval_run = _make_eval_run(
            db, user_api_key, status="completed", score=_good_score(0.72, 0.72)
        )
        history = [
            {
                "round_number": 1,
                "eval_run_id": 101,
                "config_version": 1,
                "stop_score": 0.60,
                "kb_score": None,
            },
            {
                "round_number": 2,
                "eval_run_id": 102,
                "config_version": 2,
                "stop_score": 0.70,
                "kb_score": None,
            },
            {
                "round_number": 3,
                "eval_run_id": 103,
                "config_version": 3,
                "stop_score": 0.715,
                "kb_score": None,
            },
        ]
        state = _base_state(
            current_eval_run_id=eval_run.id,
            round_number=4,
            max_rounds=10,
            history=history,
            consecutive_low_delta_rounds=2,
            best_stop_score=0.715,
            best_round_number=3,
            best_config_version=3,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        with _patch_session(db):
            result = wait_eval_node(state)

        assert result["consecutive_low_delta_rounds"] == 3
        assert result["stop_reason"] == STOP_REASON_CEILING_REACHED

    def test_reaching_max_rounds_sets_max_rounds_reached(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        # A big score jump resets consecutive_low_delta_rounds to 0 — only the
        # round-cap should trigger the stop here, not the ceiling.
        eval_run = _make_eval_run(
            db, user_api_key, status="completed", score=_good_score(0.95, 0.95)
        )
        history = [
            {
                "round_number": 1,
                "eval_run_id": 101,
                "config_version": 1,
                "stop_score": 0.40,
                "kb_score": None,
            },
            {
                "round_number": 2,
                "eval_run_id": 102,
                "config_version": 2,
                "stop_score": 0.45,
                "kb_score": None,
            },
        ]
        state = _base_state(
            current_eval_run_id=eval_run.id,
            round_number=3,
            max_rounds=3,
            history=history,
            consecutive_low_delta_rounds=1,
            best_stop_score=0.45,
            best_round_number=2,
            best_config_version=2,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        with _patch_session(db):
            result = wait_eval_node(state)

        assert result["consecutive_low_delta_rounds"] == 0
        assert result["stop_reason"] == STOP_REASON_MAX_ROUNDS_REACHED


class TestRouteAfterEval:
    def test_routes_to_finalize_node_when_stop_reason_is_set(self) -> None:
        assert (
            route_after_eval({"stop_reason": STOP_REASON_CEILING_REACHED})
            == "finalize_node"
        )

    def test_routes_to_start_improve_node_when_no_stop_reason(self) -> None:
        assert route_after_eval({"stop_reason": None}) == "start_improve_node"


class TestWaitImproveNodeBranches:
    def test_failed_job_sets_round_failed(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        job = JobCrud(session=db).create(
            job_type=JobType.PROMPT_IMPROVEMENT, project_id=user_api_key.project_id
        )
        JobCrud(session=db).update(
            job.id, JobUpdate(status=JobStatus.FAILED, error_message="llm down")
        )
        state = _base_state(
            current_improvement_job_id=str(job.id),
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        with _patch_session(db):
            result = wait_improve_node(state)

        assert result["stop_reason"] == STOP_REASON_ROUND_FAILED
        assert "llm down" in result["error_message"]

    def test_success_job_advances_round_and_config_version(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        job = JobCrud(session=db).create(
            job_type=JobType.PROMPT_IMPROVEMENT, project_id=user_api_key.project_id
        )
        JobCrud(session=db).update(
            job.id, JobUpdate(status=JobStatus.SUCCESS, meta={"version": 7})
        )
        state = _base_state(
            current_improvement_job_id=str(job.id),
            round_number=2,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        with _patch_session(db):
            result = wait_improve_node(state)

        assert result["round_number"] == 3
        assert result["config_version"] == 7
        assert result["current_improvement_job_id"] is None


class TestFinalizeNode:
    def test_ceiling_reached_persists_completed_status(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        iteration_run = _make_iteration_run(db, user_api_key)
        state = _base_state(
            iteration_run_id=iteration_run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            stop_reason=STOP_REASON_CEILING_REACHED,
            history=[
                {
                    "round_number": 1,
                    "eval_run_id": 11,
                    "config_version": 1,
                    "stop_score": 0.7,
                    "kb_score": None,
                },
            ],
            best_round_number=1,
        )

        with _patch_session(db), patch(
            "app.services.evaluations.iteration_graph.send_callback"
        ) as mock_send:
            finalize_node(state)

        db.expire_all()
        persisted = db.get(EvaluationIterationRun, iteration_run.id)
        assert persisted.status == EvaluationIterationStatusEnum.COMPLETED
        assert persisted.stop_reason == STOP_REASON_CEILING_REACHED
        mock_send.assert_called_once()

    def test_round_failed_persists_failed_status_with_error_message(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        iteration_run = _make_iteration_run(db, user_api_key)
        state = _base_state(
            iteration_run_id=iteration_run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            stop_reason=STOP_REASON_ROUND_FAILED,
            error_message="round blew up",
            history=[],
        )

        with _patch_session(db), patch(
            "app.services.evaluations.iteration_graph.send_callback"
        ):
            finalize_node(state)

        db.expire_all()
        persisted = db.get(EvaluationIterationRun, iteration_run.id)
        assert persisted.status == EvaluationIterationStatusEnum.FAILED
        assert persisted.error_message == "round blew up"

    def test_callback_payload_shape_and_best_round_is_highest_score_not_last(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        iteration_run = _make_iteration_run(db, user_api_key)
        history = [
            {
                "round_number": 1,
                "eval_run_id": 11,
                "config_version": 1,
                "stop_score": 0.60,
                "kb_score": None,
            },
            {
                "round_number": 2,
                "eval_run_id": 12,
                "config_version": 2,
                "stop_score": 0.92,
                "kb_score": 0.5,
            },
            {
                "round_number": 3,
                "eval_run_id": 13,
                "config_version": 3,
                "stop_score": 0.55,
                "kb_score": None,
            },
        ]
        state = _base_state(
            iteration_run_id=iteration_run.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            stop_reason=STOP_REASON_MAX_ROUNDS_REACHED,
            history=history,
            best_round_number=2,
            callback_url=_CALLBACK_URL,
        )

        with _patch_session(db), patch(
            "app.services.evaluations.iteration_graph.send_callback"
        ) as mock_send:
            finalize_node(state)

        mock_send.assert_called_once()
        args, kwargs = mock_send.call_args
        assert args[0] == _CALLBACK_URL
        envelope = args[1]
        report = envelope["data"]

        assert report["iteration_run_id"] == iteration_run.id
        assert report["stop_reason"] == STOP_REASON_MAX_ROUNDS_REACHED
        assert len(report["history"]) == 3
        # The best round is round 2 (highest stop_score), not round 3 (the last).
        assert report["best_round"]["round_number"] == 2
        assert report["best_round"]["round_number"] != history[-1]["round_number"]
        assert report["best_round"]["stop_score"] == pytest.approx(0.92)


class TestExecuteEvaluationIterationGraphStep:
    def test_uncaught_exception_marks_thin_row_failed_and_reraises(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        iteration_run = _make_iteration_run(db, user_api_key)

        with (
            _patch_session(db),
            patch(
                "app.services.evaluations.iteration_graph._run_graph_step",
                side_effect=RuntimeError("graph blew up"),
            ),
        ):
            with pytest.raises(RuntimeError, match="graph blew up"):
                execute_evaluation_iteration_graph_step(
                    iteration_run_id=iteration_run.id,
                    resume=False,
                    organization_id=user_api_key.organization_id,
                    project_id=user_api_key.project_id,
                )

        db.expire_all()
        persisted = db.get(EvaluationIterationRun, iteration_run.id)
        assert persisted.status == EvaluationIterationStatusEnum.FAILED
        assert (
            persisted.error_message == "Evaluation iteration loop failed unexpectedly."
        )
