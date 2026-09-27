"""Tests for the pipeline orchestrator failure guard (no dangling runs)."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded

from app.crud.assessment import core as assessment_core
from app.crud.assessment.core import _read_exec
from app.models.assessment import Stage
from app.services.assessment import tasks


@contextmanager
def _session_cm(session):
    yield session


def _patch_session(run):
    session = MagicMock()
    session.get.return_value = run
    cm = patch.object(tasks, "Session", return_value=_session_cm(session))
    return cm, session


class TestMarkRunFailed:
    def test_marks_non_terminal_run_failed(self) -> None:
        run = SimpleNamespace(
            execution={
                "stage": Stage.PRE_FILTER_TOPIC_RELEVANCE,
                "stage_status": "PENDING",
            },
            assessment_id=7,
        )
        cm, session = _patch_session(run)
        with cm, patch.object(assessment_core, "flag_modified"), patch.object(
            tasks, "update_assessment_run_status"
        ) as upd, patch.object(tasks, "recompute_assessment_status") as recompute:
            tasks._mark_run_failed(11, "boom")
        upd.assert_called_once()
        assert upd.call_args.kwargs["status"] == "failed"
        # Failed stage preserved for resume; only stage_status flips to FAILED.
        assert _read_exec(run).get("stage") == Stage.PRE_FILTER_TOPIC_RELEVANCE
        assert _read_exec(run).get("stage_status") == "FAILED"
        recompute.assert_called_once_with(session=session, assessment_id=7)

    def test_skips_terminal_run(self) -> None:
        run = SimpleNamespace(execution={"stage": Stage.COMPLETED}, assessment_id=7)
        cm, _ = _patch_session(run)
        with cm, patch.object(tasks, "update_assessment_run_status") as upd:
            tasks._mark_run_failed(11, "boom")
        upd.assert_not_called()

    def test_missing_run_noop(self) -> None:
        cm, _ = _patch_session(None)
        with cm, patch.object(tasks, "update_assessment_run_status") as upd:
            tasks._mark_run_failed(11, "boom")
        upd.assert_not_called()


class TestExecutePipelineGuard:
    def test_soft_timeout_marks_failed_and_reraises(self) -> None:
        with patch.object(
            tasks, "_orchestrate", side_effect=SoftTimeLimitExceeded()
        ), patch.object(tasks, "_mark_run_failed") as mark:
            with pytest.raises(SoftTimeLimitExceeded):
                tasks.execute_assessment_pipeline(11, 1, 1)
        mark.assert_called_once()
        assert mark.call_args.args[0] == 11

    def test_unexpected_exception_marks_failed_and_reraises(self) -> None:
        with patch.object(
            tasks, "_orchestrate", side_effect=RuntimeError("kaboom")
        ), patch.object(tasks, "_mark_run_failed") as mark:
            with pytest.raises(RuntimeError):
                tasks.execute_assessment_pipeline(11, 1, 1)
        mark.assert_called_once_with(11, "Assessment run failed unexpectedly.")

    def test_success_does_not_mark_failed(self) -> None:
        with patch.object(tasks, "_orchestrate", return_value=None), patch.object(
            tasks, "_mark_run_failed"
        ) as mark:
            tasks.execute_assessment_pipeline(11, 1, 1)
        mark.assert_not_called()
