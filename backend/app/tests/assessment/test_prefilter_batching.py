"""Tests for the single pipeline orchestrator (state-machine submit step)."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models.assessment import Stage, StageStatus
from app.services.assessment import tasks


@contextmanager
def _session_cm(session):
    yield session


def _run(**kw):
    base = {
        "id": 5,
        "assessment_id": 9,
        "input": {
            "prefilter_config": {"topic_relevance": {"columns": ["a"], "prompt": "p"}}
        },
        "config_id": "c",
        "config_version": 1,
        "pipeline": None,
        "stage": None,
        "stage_status": None,
        "status": "pending",
        "stage_batches": None,
        "total_items": 0,
    }
    base.update(kw)
    return SimpleNamespace(**base)


class TestOrchestrate:
    def test_inits_pipeline_and_submits_first_stage(self) -> None:
        run = _run()
        session = MagicMock()
        session.get.return_value = run
        with patch.object(
            tasks, "Session", return_value=_session_cm(session)
        ), patch.object(tasks, "flag_modified"), patch.object(
            tasks, "_submit_stage"
        ) as submit:
            tasks._orchestrate(5, 1, 1)
        assert run.stage == Stage.PRE_FILTER_TOPIC_RELEVANCE
        assert run.stage_status == StageStatus.PENDING
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
        return [
            patch.object(
                tasks,
                "_resolve_run_context",
                return_value=(SimpleNamespace(), MagicMock(), SimpleNamespace(), None),
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
        with p[0], p[1], p[2], p[3], patch.object(tasks, "flag_modified"), patch.object(
            tasks, "build_prefilter_requests", return_value=[{"key": "tr_0"}]
        ), patch.object(tasks, "submit_prefilter_batch", return_value=batch_job):
            tasks._submit_stage(session, run, 1, 1)
        assert run.stage_batches[Stage.PRE_FILTER_TOPIC_RELEVANCE] == 7
        assert run.stage_status == StageStatus.PROCESSING

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
