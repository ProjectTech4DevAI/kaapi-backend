"""Tests for assessment/cron.py helper functions."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.crud.assessment.cron import (
    _log_config_progress,
    poll_all_pending_assessment_evaluations,
)


def _make_assessment(*, id: int = 1, status: str = "processing") -> MagicMock:
    a = MagicMock()
    a.id = id
    a.status = status
    a.experiment_name = "exp"
    return a


def _make_run(
    *,
    id: int = 10,
    assessment_id: int = 1,
    config_id=None,
    config_version: int | None = 1,
    updated_at=None,
) -> MagicMock:
    r = MagicMock()
    r.id = id
    r.assessment_id = assessment_id
    r.config_id = config_id
    r.config_version = config_version
    r.updated_at = updated_at or datetime(2024, 6, 1, 12, 0, 0)
    return r


class TestLogConfigProgress:
    def test_no_log_for_no_change_action(self) -> None:
        run = _make_run()
        assessment = _make_assessment()
        assert _log_config_progress({"action": "no_change"}, run, assessment) is None

    def test_no_log_for_still_processing(self) -> None:
        run = _make_run()
        assessment = _make_assessment()
        assert (
            _log_config_progress({"action": "still_processing"}, run, assessment)
            is None
        )

    def test_processed_action_does_not_raise(self) -> None:
        run = _make_run()
        assessment = _make_assessment()
        _log_config_progress(
            {
                "action": "processed",
                "current_status": "completed",
                "provider_status": "completed",
            },
            run,
            assessment,
        )

    def test_failed_action_does_not_raise(self) -> None:
        run = _make_run()
        assessment = _make_assessment()
        _log_config_progress(
            {
                "action": "failed",
                "current_status": "failed",
                "provider_status": "failed",
            },
            run,
            assessment,
        )


class TestPollAllPendingAssessmentEvaluations:
    @pytest.mark.asyncio
    async def test_no_pending_assessments(self) -> None:
        session = MagicMock()
        session.exec.return_value.all.return_value = []
        result = await poll_all_pending_assessment_evaluations(session=session)
        assert result["total"] == 0
        assert result["processed"] == 0

    @pytest.mark.asyncio
    async def test_no_active_runs_recompute(self) -> None:
        session = MagicMock()
        assessment = _make_assessment(id=1, status="processing")
        session.exec.return_value.all.return_value = [assessment]
        refreshed = _make_assessment(id=1, status="processing")

        run = _make_run(id=11, config_version=1)
        run.status = "completed"

        with patch(
            "app.crud.assessment.cron.get_assessment_runs_for_assessment",
            return_value=[run],
        ), patch(
            "app.crud.assessment.cron.recompute_assessment_status",
            return_value=refreshed,
        ), patch(
            "app.crud.assessment.cron.process_run_batches", new=AsyncMock()
        ):
            result = await poll_all_pending_assessment_evaluations(session=session)

        assert result["total"] == 1
        assert result["still_processing"] == 1

    @pytest.mark.asyncio
    async def test_active_run_processed(self) -> None:
        session = MagicMock()
        assessment = _make_assessment(id=1, status="processing")
        run = _make_run(id=11)
        run.stage_status = "PROCESSING"
        session.exec.return_value.all.return_value = [assessment]

        with patch(
            "app.crud.assessment.cron.get_assessment_runs_for_assessment",
            return_value=[run],
        ), patch(
            "app.crud.assessment.cron.process_run_batches",
            new=AsyncMock(
                return_value={
                    "action": "processed",
                    "current_status": "completed",
                    "provider_status": "completed",
                }
            ),
        ):
            result = await poll_all_pending_assessment_evaluations(session=session)

        assert result["processed"] == 1

    @pytest.mark.asyncio
    async def test_transient_poll_exception_does_not_fail_run(self) -> None:
        """A transient error while polling leaves the run active for retry."""
        session = MagicMock()
        assessment = _make_assessment(id=1, status="processing")
        run = _make_run(id=11)
        run.stage_status = "PROCESSING"
        session.exec.return_value.all.return_value = [assessment]

        with patch(
            "app.crud.assessment.cron.get_assessment_runs_for_assessment",
            return_value=[run],
        ), patch(
            "app.crud.assessment.cron.process_run_batches",
            new=AsyncMock(side_effect=RuntimeError("nodename nor servname provided")),
        ):
            result = await poll_all_pending_assessment_evaluations(session=session)

        assert result["failed"] == 0
        assert result["still_processing"] == 1

    @pytest.mark.asyncio
    async def test_deterministic_error_marks_run_failed(self) -> None:
        """A deterministic ValueError fails the run instead of retrying forever."""
        session = MagicMock()
        assessment = _make_assessment(id=1, status="processing")
        run = _make_run(id=11)
        run.stage_status = "PROCESSING"
        session.exec.return_value.all.return_value = [assessment]

        with patch(
            "app.crud.assessment.cron.get_assessment_runs_for_assessment",
            return_value=[run],
        ), patch(
            "app.crud.assessment.cron.process_run_batches",
            new=AsyncMock(side_effect=ValueError("Parent assessment 1 not found")),
        ), patch(
            "app.crud.assessment.cron.update_assessment_run_status"
        ) as mark_failed:
            result = await poll_all_pending_assessment_evaluations(session=session)

        assert result["failed"] == 1
        assert result["still_processing"] == 0
        assert mark_failed.call_args.kwargs["status"] == "failed"
