"""Tests for assessment/cron.py helper functions."""

from unittest.mock import MagicMock

import pytest

from app.assessment.cron import (
    _build_callback_payload,
    _log_config_progress,
    poll_all_pending_assessment_evaluations,
)


def _make_assessment(*, id: int = 1, status: str = "processing") -> MagicMock:
    a = MagicMock()
    a.id = id
    a.status = status
    return a


def _make_run(
    *,
    id: int = 10,
    run_name: str = "exp_v1",
    assessment_id: int = 1,
    config_id=None,
    config_version: int | None = 1,
    updated_at=None,
) -> MagicMock:
    from datetime import datetime

    r = MagicMock()
    r.id = id
    r.run_name = run_name
    r.assessment_id = assessment_id
    r.config_id = config_id
    r.config_version = config_version
    r.updated_at = updated_at or datetime(2024, 6, 1, 12, 0, 0)
    return r


class TestLogConfigProgress:
    def test_no_log_for_no_change_action(self) -> None:
        run = _make_run()
        # should not raise and should return None
        assert _log_config_progress({"action": "no_change"}, run) is None

    def test_no_log_for_still_processing(self) -> None:
        run = _make_run()
        assert _log_config_progress({"action": "still_processing"}, run) is None

    def test_processed_action_does_not_raise(self) -> None:
        run = _make_run()
        _log_config_progress(
            {
                "action": "processed",
                "current_status": "completed",
                "provider_status": "completed",
            },
            run,
        )

    def test_failed_action_does_not_raise(self) -> None:
        run = _make_run()
        _log_config_progress(
            {
                "action": "failed",
                "current_status": "failed",
                "provider_status": "failed",
            },
            run,
        )


class TestBuildCallbackPayload:
    def test_type_at_root(self) -> None:
        assessment = _make_assessment()
        run = _make_run()
        payload = _build_callback_payload(
            assessment, run, {"current_status": "completed"}
        )
        # events.py reads payload.get("type") — must be at root
        assert payload.get("type") == "assessment.child_status_changed"

    def test_type_also_inside_data(self) -> None:
        assessment = _make_assessment()
        run = _make_run()
        payload = _build_callback_payload(
            assessment, run, {"current_status": "completed"}
        )
        assert payload["data"]["type"] == "assessment.child_status_changed"

    def test_assessment_info_in_data(self) -> None:
        assessment = _make_assessment(id=5, status="processing")
        run = _make_run(id=99, config_version=3)
        payload = _build_callback_payload(
            assessment, run, {"current_status": "failed", "error": "timeout"}
        )
        data = payload["data"]
        assert data["assessment_id"] == 5
        assert data["assessment_status"] == "processing"
        assert data["run"]["id"] == 99
        assert data["run"]["config_version"] == 3
        assert data["run"]["status"] == "failed"
        assert data["run"]["error"] == "timeout"

    def test_run_updated_at_serialized(self) -> None:
        from datetime import datetime

        assessment = _make_assessment()
        run = _make_run(updated_at=datetime(2024, 3, 15, 10, 30, 0))
        payload = _build_callback_payload(assessment, run, {})
        assert payload["data"]["run"]["updated_at"] == "2024-03-15T10:30:00"

    def test_config_id_none_stays_none(self) -> None:
        assessment = _make_assessment()
        run = _make_run(config_id=None)
        payload = _build_callback_payload(assessment, run, {})
        assert payload["data"]["run"]["config_id"] is None

    def test_response_is_success(self) -> None:
        assessment = _make_assessment()
        run = _make_run()
        payload = _build_callback_payload(assessment, run, {})
        assert payload["success"] is True


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

        from unittest.mock import AsyncMock, patch

        with patch(
            "app.assessment.cron.get_assessment_runs_for_manager",
            return_value=[_make_run(id=11, run_name="exp", config_version=1)],
        ), patch(
            "app.assessment.cron.recompute_assessment_status",
            return_value=refreshed,
        ), patch("app.assessment.cron.check_and_process_assessment", new=AsyncMock()):
            result = await poll_all_pending_assessment_evaluations(session=session)

        assert result["total"] == 1
        assert result["still_processing"] == 1

    @pytest.mark.asyncio
    async def test_active_run_processed_publishes_event(self) -> None:
        session = MagicMock()
        assessment = _make_assessment(id=1, status="processing")
        run = _make_run(id=11)
        run.status = "processing"
        session.exec.return_value.all.return_value = [assessment]
        session.get.return_value = assessment

        from unittest.mock import AsyncMock, patch

        with patch(
            "app.assessment.cron.get_assessment_runs_for_manager",
            return_value=[run],
        ), patch(
            "app.assessment.cron.check_and_process_assessment",
            new=AsyncMock(
                return_value={
                    "action": "processed",
                    "current_status": "completed",
                    "provider_status": "completed",
                }
            ),
        ), patch("app.assessment.cron.assessment_event_broker.publish") as publish:
            result = await poll_all_pending_assessment_evaluations(session=session)

        assert result["processed"] == 1
        publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_active_run_failure_and_cleanup_failure(self) -> None:
        session = MagicMock()
        assessment = _make_assessment(id=1, status="processing")
        run = _make_run(id=11)
        run.status = "processing"
        session.exec.return_value.all.return_value = [assessment]

        from unittest.mock import AsyncMock, patch

        with patch(
            "app.assessment.cron.get_assessment_runs_for_manager",
            return_value=[run],
        ), patch(
            "app.assessment.cron.check_and_process_assessment",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ), patch(
            "app.assessment.cron.update_assessment_run_status",
            side_effect=RuntimeError("cleanup-failed"),
        ), patch(
            "app.assessment.cron.recompute_assessment_status",
        ):
            result = await poll_all_pending_assessment_evaluations(session=session)

        assert result["failed"] == 1
