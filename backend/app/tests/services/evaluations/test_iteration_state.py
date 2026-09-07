"""Round bookkeeping extracted from `wait_eval_node` (`iteration_state.py`).

`advance_round_state` decides when the loop stops, so the ceiling counter's reset
rule and the max-rounds cap are the contract worth pinning. Pure — no graph, no DB.
"""

from typing import Any

import pytest

from app.core.config import settings
from app.models.evaluation_iteration import EvaluationIterationStatusEnum
from app.services.evaluations.iteration import (
    STOP_REASON_CEILING_REACHED,
    STOP_REASON_MAX_ROUNDS_REACHED,
)
from app.services.evaluations.iteration_state import (
    EvaluationIterationState,
    advance_round_state,
    build_iteration_report,
)

_BIG_GAIN = settings.EVAL_ITERATION_CEILING_DELTA_THRESHOLD * 10
_SMALL_GAIN = settings.EVAL_ITERATION_CEILING_DELTA_THRESHOLD / 10


def _state(**overrides: Any) -> EvaluationIterationState:
    base: dict[str, Any] = {
        "iteration_run_id": 1,
        "dataset_id": 2,
        "experiment_name": "exp",
        "config_id": "00000000-0000-0000-0000-000000000000",
        "config_version": 1,
        "round_number": 1,
        "max_rounds": 10,
        "current_eval_run_id": 100,
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
        "callback_url": "https://example.com/hook",
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


def _round(round_number: int, stop_score: float) -> dict[str, Any]:
    return {
        "round_number": round_number,
        "eval_run_id": 100 + round_number,
        "config_version": round_number,
        "stop_score": stop_score,
        "kb_score": None,
    }


class TestAdvanceRoundState:
    def test_first_round_records_history_and_becomes_best(self) -> None:
        update = advance_round_state(
            state=_state(), eval_run_id=100, stop_score=3.0, kb_score=2.0
        )
        assert update["history"] == [
            {
                "round_number": 1,
                "eval_run_id": 100,
                "config_version": 1,
                "stop_score": 3.0,
                "kb_score": 2.0,
            }
        ]
        assert update["best_stop_score"] == 3.0
        assert update["best_round_number"] == 1
        assert update["best_config_version"] == 1

    def test_first_round_never_counts_as_low_delta(self) -> None:
        update = advance_round_state(
            state=_state(), eval_run_id=100, stop_score=0.0, kb_score=None
        )
        assert update["consecutive_low_delta_rounds"] == 0
        assert "stop_reason" not in update

    def test_a_worse_round_leaves_best_untouched(self) -> None:
        state = _state(
            round_number=2,
            config_version=2,
            history=[_round(1, 4.0)],
            best_stop_score=4.0,
            best_round_number=1,
            best_config_version=1,
        )
        update = advance_round_state(
            state=state, eval_run_id=101, stop_score=1.0, kb_score=None
        )
        assert update["best_stop_score"] == 4.0
        assert update["best_round_number"] == 1
        assert update["best_config_version"] == 1

    def test_a_better_round_takes_over_best(self) -> None:
        state = _state(
            round_number=2,
            config_version=2,
            history=[_round(1, 1.0)],
            best_stop_score=1.0,
            best_round_number=1,
            best_config_version=1,
        )
        update = advance_round_state(
            state=state, eval_run_id=101, stop_score=4.0, kb_score=None
        )
        assert update["best_stop_score"] == 4.0
        assert update["best_round_number"] == 2
        assert update["best_config_version"] == 2

    def test_a_gain_below_threshold_increments_the_ceiling_counter(self) -> None:
        state = _state(
            round_number=2, history=[_round(1, 3.0)], consecutive_low_delta_rounds=1
        )
        update = advance_round_state(
            state=state, eval_run_id=101, stop_score=3.0 + _SMALL_GAIN, kb_score=None
        )
        assert update["consecutive_low_delta_rounds"] == 2

    def test_a_gain_at_or_above_threshold_resets_the_counter(self) -> None:
        state = _state(
            round_number=2, history=[_round(1, 3.0)], consecutive_low_delta_rounds=2
        )
        update = advance_round_state(
            state=state, eval_run_id=101, stop_score=3.0 + _BIG_GAIN, kb_score=None
        )
        assert update["consecutive_low_delta_rounds"] == 0
        assert "stop_reason" not in update

    def test_a_regression_counts_as_low_delta(self) -> None:
        state = _state(round_number=2, history=[_round(1, 4.0)])
        update = advance_round_state(
            state=state, eval_run_id=101, stop_score=1.0, kb_score=None
        )
        assert update["consecutive_low_delta_rounds"] == 1

    def test_hitting_the_consecutive_cap_stops_on_ceiling(self) -> None:
        cap = settings.EVAL_ITERATION_CEILING_CONSECUTIVE_ROUNDS
        state = _state(
            round_number=cap + 1,
            history=[_round(1, 3.0)],
            consecutive_low_delta_rounds=cap - 1,
        )
        update = advance_round_state(
            state=state, eval_run_id=101, stop_score=3.0, kb_score=None
        )
        assert update["consecutive_low_delta_rounds"] == cap
        assert update["stop_reason"] == STOP_REASON_CEILING_REACHED

    def test_the_last_allowed_round_stops_on_max_rounds(self) -> None:
        state = _state(round_number=3, max_rounds=3, history=[_round(1, 1.0)])
        update = advance_round_state(
            state=state, eval_run_id=101, stop_score=1.0 + _BIG_GAIN, kb_score=None
        )
        assert update["stop_reason"] == STOP_REASON_MAX_ROUNDS_REACHED

    def test_ceiling_wins_over_max_rounds_when_both_apply(self) -> None:
        cap = settings.EVAL_ITERATION_CEILING_CONSECUTIVE_ROUNDS
        state = _state(
            round_number=3,
            max_rounds=3,
            history=[_round(1, 3.0)],
            consecutive_low_delta_rounds=cap - 1,
        )
        update = advance_round_state(
            state=state, eval_run_id=101, stop_score=3.0, kb_score=None
        )
        assert update["stop_reason"] == STOP_REASON_CEILING_REACHED

    def test_kb_score_is_recorded_but_never_gates_stopping(self) -> None:
        update = advance_round_state(
            state=_state(), eval_run_id=100, stop_score=5.0, kb_score=0.0
        )
        assert update["history"][0]["kb_score"] == 0.0
        assert "stop_reason" not in update


class TestBuildIterationReport:
    def test_report_carries_history_and_resolves_the_best_round(self) -> None:
        state = _state(
            history=[_round(1, 1.0), _round(2, 4.0)],
            best_round_number=2,
            stop_reason=STOP_REASON_CEILING_REACHED,
        )
        report = build_iteration_report(state, EvaluationIterationStatusEnum.COMPLETED)
        assert report.iteration_run_id == 1
        assert report.status is EvaluationIterationStatusEnum.COMPLETED
        assert report.stop_reason == STOP_REASON_CEILING_REACHED
        assert [r.round_number for r in report.history] == [1, 2]
        assert report.best_round is not None
        assert report.best_round.round_number == 2

    def test_best_round_is_none_when_no_round_completed(self) -> None:
        report = build_iteration_report(
            _state(error_message="boom"), EvaluationIterationStatusEnum.FAILED
        )
        assert report.best_round is None
        assert report.history == []
        assert report.error_message == "boom"

    @pytest.mark.parametrize(
        "status",
        [
            EvaluationIterationStatusEnum.COMPLETED,
            EvaluationIterationStatusEnum.FAILED,
        ],
    )
    def test_status_is_passed_through(
        self, status: EvaluationIterationStatusEnum
    ) -> None:
        assert build_iteration_report(_state(), status).status is status
