"""Tests for the single pipeline orchestrator (state-machine submit step)."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded

from app.crud.assessment import core as assessment_core
from app.crud.assessment.core import _read_exec
from app.models.assessment import AssessmentRun, Stage, StageStatus
from app.services.assessment import tasks

# Runtime state that used to be its own AssessmentRun column now lives in the exec bag.
_EXEC_BAG_KEYS = (
    "stage",
    "stage_status",
    "pipeline",
    "stage_batches",
    "accepted_indices",
)

# The run input binding moved to the parent assessment; tests that reach the
# parent-assessment load hand this to an Assessment stub via ``session.get``.
_ASSESSMENT_INPUT = {
    "prefilter_config": {"topic_relevance": {"columns": ["a"], "prompt": "p"}}
}


@contextmanager
def _session_cm(session):
    yield session


def _run(**kw):
    execution = {key: kw.pop(key) for key in _EXEC_BAG_KEYS if key in kw}
    kw.pop("input", None)  # run input now lives on the parent assessment
    base = {
        "id": 5,
        "assessment_id": 9,
        "config_id": "c",
        "config_version": 1,
        "status": "pending",
        "total_items": 0,
    }
    base.update(kw)
    base["execution"] = execution
    return SimpleNamespace(**base)


class TestOrchestrate:
    def test_inits_pipeline_and_submits_first_stage(self) -> None:
        run = _run()
        assessment = SimpleNamespace(input=_ASSESSMENT_INPUT)
        session = MagicMock()
        session.get.side_effect = lambda model, _id: (
            run if model is AssessmentRun else assessment
        )
        with patch.object(
            tasks, "Session", return_value=_session_cm(session)
        ), patch.object(assessment_core, "flag_modified"), patch.object(
            tasks, "_submit_stage"
        ) as submit:
            tasks._orchestrate(5, 1, 1)
        assert _read_exec(run).get("stage") == Stage.PRE_FILTER_TOPIC_RELEVANCE
        assert _read_exec(run).get("stage_status") == StageStatus.PENDING
        submit.assert_called_once()

    def test_skips_when_not_pending(self) -> None:
        run = _run(
            pipeline={"stages": [{"stage": Stage.L2_ASSESSMENT, "order": 1}]},
            stage=Stage.L2_ASSESSMENT,
            stage_status=StageStatus.PROCESSING,
        )
        session = MagicMock()
        session.get.return_value = run
        with patch.object(
            tasks, "Session", return_value=_session_cm(session)
        ), patch.object(tasks, "_submit_stage") as submit:
            tasks._orchestrate(5, 1, 1)
        submit.assert_not_called()

    def test_terminal_stage_returns(self) -> None:
        run = _run(stage=Stage.COMPLETED)
        session = MagicMock()
        session.get.return_value = run
        with patch.object(
            tasks, "Session", return_value=_session_cm(session)
        ), patch.object(tasks, "_submit_stage") as submit:
            tasks._orchestrate(5, 1, 1)
        submit.assert_not_called()


class TestSubmitCurrentStage:
    def _ctx(self, accepted):
        assessment = SimpleNamespace(input=_ASSESSMENT_INPUT)
        return [
            patch.object(
                tasks,
                "_resolve_run_context",
                return_value=(assessment, MagicMock(), SimpleNamespace(), None),
            ),
            patch.object(tasks, "_load_dataset_rows", return_value=[{"a": "1"}] * 3),
            patch.object(tasks, "_accepted_indices", return_value=accepted),
            patch.object(tasks, "recompute_assessment_status"),
        ]

    def test_submits_prefilter_batch(self) -> None:
        run = _run(
            stage=Stage.PRE_FILTER_TOPIC_RELEVANCE,
            stage_status=StageStatus.PENDING,
            stage_batches={},
        )
        session = MagicMock()
        batch_job = SimpleNamespace(id=7, total_items=3)
        p = self._ctx([0, 1, 2])
        with p[0], p[1], p[2], p[3], patch.object(
            assessment_core, "flag_modified"
        ), patch.object(
            tasks, "build_prefilter_requests", return_value=[{"key": "tr_0"}]
        ), patch.object(
            tasks, "submit_prefilter_batch", return_value=batch_job
        ):
            tasks._submit_stage(session, run, 1, 1)
        assert (
            _read_exec(run).get("stage_batches")[Stage.PRE_FILTER_TOPIC_RELEVANCE] == 7
        )
        assert _read_exec(run).get("stage_status") == StageStatus.PROCESSING

    def test_zero_accepted_advances(self) -> None:
        run = _run(
            stage=Stage.L2_ASSESSMENT,
            stage_status=StageStatus.PENDING,
            stage_batches={},
        )
        session = MagicMock()
        p = self._ctx([])
        with p[0], p[1], p[2], p[3], patch.object(tasks, "_persist_advance") as advance:
            tasks._submit_stage(session, run, 1, 1)
        advance.assert_called_once()


class TestAcceptedIndices:
    def test_uses_persisted_indices_without_downloading(self) -> None:
        """Stored accepted set is read directly — no gate batch re-download."""
        run = _run(
            pipeline={
                "stages": [
                    {"stage": Stage.PRE_FILTER_TOPIC_RELEVANCE, "order": 1},
                    {"stage": Stage.L2_ASSESSMENT, "order": 2},
                ],
                "accepted_indices": [0, 2, 5],
            },
            stage=Stage.L2_ASSESSMENT,
        )
        with patch.object(tasks, "load_raw_batch_results") as load:
            result = tasks._accepted_indices(
                MagicMock(), run, total_rows=10, project_id=1
            )
        assert result == [0, 2, 5]
        load.assert_not_called()

    def test_persisted_indices_clamped_to_total_rows(self) -> None:
        run = _run(
            pipeline={"stages": [], "accepted_indices": [0, 3, 99]},
            stage=Stage.L2_ASSESSMENT,
        )
        result = tasks._accepted_indices(MagicMock(), run, total_rows=4, project_id=1)
        assert result == [0, 3]

    def test_falls_back_to_full_range_when_nothing_persisted(self) -> None:
        run = _run(
            pipeline={"stages": [{"stage": Stage.L2_ASSESSMENT, "order": 1}]},
            stage=Stage.L2_ASSESSMENT,
        )
        result = tasks._accepted_indices(MagicMock(), run, total_rows=3, project_id=1)
        assert result == [0, 1, 2]


class TestGuardEntrypoint:
    def test_unexpected_exception_marks_failed_and_reraises(self) -> None:
        with patch.object(
            tasks, "_orchestrate", side_effect=RuntimeError("boom")
        ), patch.object(tasks, "_mark_run_failed") as mark:
            with pytest.raises(RuntimeError):
                tasks.execute_assessment_pipeline(5, 1, 1)
        mark.assert_called_once()

    def test_soft_timeout_marks_failed_and_reraises(self) -> None:
        with patch.object(
            tasks, "_orchestrate", side_effect=SoftTimeLimitExceeded()
        ), patch.object(tasks, "_mark_run_failed") as mark:
            with pytest.raises(SoftTimeLimitExceeded):
                tasks.execute_assessment_pipeline(5, 1, 1)
        mark.assert_called_once()


class TestMarkRunFailed:
    def test_marks_non_terminal_run_failed(self) -> None:
        run = _run(stage=Stage.L2_ASSESSMENT, stage_status=StageStatus.PROCESSING)
        session = MagicMock()
        session.get.return_value = run
        with patch.object(
            tasks, "Session", return_value=_session_cm(session)
        ), patch.object(assessment_core, "flag_modified"), patch.object(
            tasks, "update_assessment_run_status"
        ) as upd, patch.object(
            tasks, "recompute_assessment_status"
        ):
            tasks._mark_run_failed(5, "dead")
        assert _read_exec(run).get("stage_status") == StageStatus.FAILED
        upd.assert_called_once()

    def test_skips_terminal_run(self) -> None:
        run = _run(stage=Stage.COMPLETED)
        session = MagicMock()
        session.get.return_value = run
        with patch.object(
            tasks, "Session", return_value=_session_cm(session)
        ), patch.object(tasks, "update_assessment_run_status") as upd:
            tasks._mark_run_failed(5, "dead")
        upd.assert_not_called()


class TestDispatch:
    def test_dispatch_enqueues_task(self) -> None:
        with patch.object(tasks, "run_assessment_pipeline") as task:
            tasks._dispatch(5, 1, 2)
        task.delay.assert_called_once()
        assert task.delay.call_args.kwargs["run_id"] == 5


class TestResolveRunContext:
    def test_success(self) -> None:
        session = MagicMock()
        run = _run()
        session.get.return_value = SimpleNamespace(dataset_id=3)
        with patch.object(
            tasks, "get_assessment_dataset_by_id", return_value=MagicMock()
        ), patch.object(
            tasks, "resolve_evaluation_config", return_value=({"x": 1}, None)
        ):
            _a, _d, blob, err = tasks._resolve_run_context(session, run, 1, 1)
        assert blob == {"x": 1}
        assert err is None

    def test_missing_parent(self) -> None:
        session = MagicMock()
        session.get.return_value = None
        _a, _d, blob, err = tasks._resolve_run_context(session, _run(), 1, 1)
        assert blob is None
        assert "not found" in err

    def test_config_error(self) -> None:
        session = MagicMock()
        session.get.return_value = SimpleNamespace(dataset_id=3)
        with patch.object(
            tasks, "get_assessment_dataset_by_id", return_value=MagicMock()
        ), patch.object(
            tasks, "resolve_evaluation_config", return_value=(None, "bad config")
        ):
            _a, _d, blob, err = tasks._resolve_run_context(session, _run(), 1, 1)
        assert blob is None
        assert "bad config" in err


class TestAcceptedIndicesFallback:
    def test_recomputes_from_gate_batch(self) -> None:
        run = _run(
            pipeline={
                "stages": [
                    {"stage": Stage.PRE_FILTER_TOPIC_RELEVANCE, "order": 1},
                    {"stage": Stage.L2_ASSESSMENT, "order": 2},
                ]
            },
            stage=Stage.L2_ASSESSMENT,
            stage_batches={Stage.PRE_FILTER_TOPIC_RELEVANCE: 1},
        )
        with patch.object(
            tasks, "get_batch_job", return_value=SimpleNamespace(provider="openai")
        ), patch.object(tasks, "load_raw_batch_results", return_value=[]), patch.object(
            tasks, "parse_assessment_output", return_value=[]
        ), patch.dict(
            tasks.STAGE_PARSERS,
            {
                Stage.PRE_FILTER_TOPIC_RELEVANCE: lambda outs: {
                    0: {"verdict": True},
                    1: {"verdict": False},
                }
            },
        ):
            result = tasks._accepted_indices(
                MagicMock(), run, total_rows=3, project_id=1
            )
        # Only row 0 passed the gate.
        assert result == [0]


class TestSubmitStageBranches:
    def test_config_error_fails_run(self) -> None:
        run = _run(stage=Stage.L2_ASSESSMENT, stage_status=StageStatus.PENDING)
        with patch.object(
            tasks, "_resolve_run_context", return_value=(None, None, None, "boom")
        ), patch.object(assessment_core, "flag_modified"), patch.object(
            tasks, "update_assessment_run_status"
        ) as upd, patch.object(
            tasks, "recompute_assessment_status"
        ):
            tasks._submit_stage(MagicMock(), run, 1, 1)
        assert _read_exec(run).get("stage_status") == StageStatus.FAILED
        upd.assert_called_once()

    def test_empty_dataset_fails_run(self) -> None:
        run = _run(stage=Stage.L2_ASSESSMENT, stage_status=StageStatus.PENDING)
        with patch.object(
            tasks,
            "_resolve_run_context",
            return_value=(SimpleNamespace(), MagicMock(), SimpleNamespace(), None),
        ), patch.object(tasks, "_load_dataset_rows", return_value=[]), patch.object(
            assessment_core, "flag_modified"
        ), patch.object(
            tasks, "update_assessment_run_status"
        ) as upd, patch.object(
            tasks, "recompute_assessment_status"
        ):
            tasks._submit_stage(MagicMock(), run, 1, 1)
        assert _read_exec(run).get("stage_status") == StageStatus.FAILED
        upd.assert_called_once()

    def test_submits_l2_batch(self) -> None:
        run = _run(
            stage=Stage.L2_ASSESSMENT,
            stage_status=StageStatus.PENDING,
            stage_batches={},
        )
        batch_job = SimpleNamespace(id=8, total_items=2)
        with patch.object(
            tasks,
            "_resolve_run_context",
            return_value=(
                SimpleNamespace(input={}),
                MagicMock(),
                SimpleNamespace(),
                None,
            ),
        ), patch.object(
            tasks, "_load_dataset_rows", return_value=[{"a": "1"}] * 3
        ), patch.object(
            tasks, "_accepted_indices", return_value=[0, 1]
        ), patch.object(
            assessment_core, "flag_modified"
        ), patch.object(
            tasks, "submit_assessment_batch", return_value=batch_job
        ), patch.object(
            tasks, "recompute_assessment_status"
        ):
            tasks._submit_stage(MagicMock(), run, 1, 1)
        assert run.total_items == 2
        assert _read_exec(run).get("stage_batches")[Stage.L2_ASSESSMENT] == 8

    def test_unknown_stage_raises(self) -> None:
        run = _run(stage="BOGUS", stage_status=StageStatus.PENDING, stage_batches={})
        with patch.object(
            tasks,
            "_resolve_run_context",
            return_value=(
                SimpleNamespace(input={}),
                MagicMock(),
                SimpleNamespace(),
                None,
            ),
        ), patch.object(
            tasks, "_load_dataset_rows", return_value=[{"a": "1"}]
        ), patch.object(
            tasks, "_accepted_indices", return_value=[0]
        ):
            with pytest.raises(ValueError):
                tasks._submit_stage(MagicMock(), run, 1, 1)


class TestPersistAdvance:
    def test_dispatches_next_stage(self) -> None:
        run = _run()
        with patch.object(
            tasks, "advance_or_finalize", return_value=Stage.L2_ASSESSMENT
        ), patch.object(tasks, "recompute_assessment_status"), patch.object(
            tasks, "_dispatch"
        ) as dispatch:
            tasks._persist_advance(MagicMock(), run, 1, 1)
        dispatch.assert_called_once()

    def test_finalize_does_not_dispatch(self) -> None:
        run = _run()
        with patch.object(
            tasks, "advance_or_finalize", return_value=None
        ), patch.object(tasks, "recompute_assessment_status"), patch.object(
            tasks, "_dispatch"
        ) as dispatch:
            tasks._persist_advance(MagicMock(), run, 1, 1)
        dispatch.assert_not_called()

    def test_enqueue_failure_marks_failed(self) -> None:
        run = _run(stage=Stage.L2_ASSESSMENT)
        with patch.object(
            tasks, "advance_or_finalize", return_value=Stage.L2_ASSESSMENT
        ), patch.object(tasks, "recompute_assessment_status"), patch.object(
            tasks, "_dispatch", side_effect=RuntimeError("broker down")
        ), patch.object(
            assessment_core, "flag_modified"
        ), patch.object(
            tasks, "update_assessment_run_status"
        ) as upd:
            tasks._persist_advance(MagicMock(), run, 1, 1)
        assert _read_exec(run).get("stage_status") == StageStatus.FAILED
        upd.assert_called_once()
