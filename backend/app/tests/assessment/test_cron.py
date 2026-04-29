"""Tests for assessment/cron.py helper functions."""

from unittest.mock import MagicMock

import pytest

from app.assessment.cron import _build_callback_payload, _log_config_progress


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
