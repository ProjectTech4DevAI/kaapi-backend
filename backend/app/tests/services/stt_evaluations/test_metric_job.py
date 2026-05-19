"""Test cases for STT automated metric computation job."""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout

from app.services.stt_evaluations.metric_job import execute_metric_computation


def _make_result(
    id: int,
    stt_sample_id: int,
    transcription: str = "hello world",
    status: str = "SUCCESS",
    evaluation_run_id: int = 1,
) -> MagicMock:
    """Create a mock STTResult."""
    result = MagicMock()
    result.id = id
    result.stt_sample_id = stt_sample_id
    result.transcription = transcription
    result.status = status
    result.evaluation_run_id = evaluation_run_id
    return result


def _make_sample(
    id: int,
    ground_truth: str | None = "hello world",
    language_id: int | None = None,
) -> MagicMock:
    """Create a mock STTSample."""
    sample = MagicMock()
    sample.id = id
    sample.ground_truth = ground_truth
    sample.language_id = language_id
    return sample


def _make_language(id: int, locale: str = "en") -> MagicMock:
    """Create a mock Language."""
    lang = MagicMock()
    lang.id = id
    lang.locale = locale
    return lang


BASE_KWARGS: dict[str, Any] = {
    "project_id": 1,
    "job_id": "10",
    "task_id": "celery-task-123",
    "task_instance": MagicMock(),
    "organization_id": 1,
    "run_id": 10,
}


@patch("app.services.stt_evaluations.metric_job.update_stt_run")
@patch("app.services.stt_evaluations.metric_job.calculate_stt_metrics")
@patch("app.services.stt_evaluations.metric_job.now")
@patch("app.services.stt_evaluations.metric_job.Session")
class TestExecuteMetricComputation:
    """Test cases for execute_metric_computation."""

    def test_no_results_returns_zero_counts(
        self, mock_session_cls, mock_now, mock_calc, mock_update_run
    ) -> None:
        """Test that no successful results returns zero counts."""
        session = mock_session_cls.return_value.__enter__.return_value
        # First query (results) returns empty
        session.exec.return_value.all.return_value = []

        result = execute_metric_computation(**BASE_KWARGS)

        assert result == {"success": True, "scored": 0, "skipped": 0, "failed": 0}
        mock_calc.assert_not_called()
        mock_update_run.assert_not_called()

    def test_scores_single_result(
        self, mock_session_cls, mock_now, mock_calc, mock_update_run
    ) -> None:
        """Test scoring a single result with matching sample and language resolution."""
        session = mock_session_cls.return_value.__enter__.return_value
        timestamp = datetime(2026, 1, 1)
        mock_now.return_value = timestamp

        stt_result = _make_result(id=1, stt_sample_id=100)
        sample = _make_sample(id=100, ground_truth="नमस्ते", language_id=5)
        language = _make_language(id=5, locale="hi")

        # Chain: results query -> samples query -> languages query
        session.exec.return_value.all.side_effect = [
            [stt_result],  # results
            [sample],  # samples
            [language],  # languages
        ]

        scores = {"wer": 0.1, "cer": 0.05, "lenient_wer": 0.08, "wip": 0.9}
        mock_calc.return_value = scores

        result = execute_metric_computation(**BASE_KWARGS)

        assert result["success"] is True
        assert result["scored"] == 1
        assert result["skipped"] == 0
        assert result["failed"] == 0

        # Verify language_id was resolved to locale and passed through
        mock_calc.assert_called_once_with(
            hypothesis="hello world",
            reference="नमस्ते",
            language_code="hi",
        )
        # Bulk update called
        session.execute.assert_called_once()
        session.commit.assert_called_once()
        # Aggregate scores stored
        mock_update_run.assert_called_once()

    def test_skips_result_without_sample(
        self, mock_session_cls, mock_now, mock_calc, mock_update_run
    ) -> None:
        """Test that results without matching samples are skipped."""
        session = mock_session_cls.return_value.__enter__.return_value
        mock_now.return_value = datetime(2026, 1, 1)

        stt_result = _make_result(id=1, stt_sample_id=999)  # no matching sample

        session.exec.return_value.all.side_effect = [
            [stt_result],  # results
            [],  # samples (empty — no match)
        ]

        result = execute_metric_computation(**BASE_KWARGS)

        assert result["success"] is True
        assert result["scored"] == 0
        assert result["skipped"] == 1
        mock_calc.assert_not_called()

    def test_skips_result_without_ground_truth(
        self, mock_session_cls, mock_now, mock_calc, mock_update_run
    ) -> None:
        """Test that results whose samples lack ground truth are skipped."""
        session = mock_session_cls.return_value.__enter__.return_value
        mock_now.return_value = datetime(2026, 1, 1)

        stt_result = _make_result(id=1, stt_sample_id=100)
        sample = _make_sample(id=100, ground_truth=None)

        session.exec.return_value.all.side_effect = [
            [stt_result],
            [sample],
        ]

        result = execute_metric_computation(**BASE_KWARGS)

        assert result["success"] is True
        assert result["scored"] == 0
        assert result["skipped"] == 1
        mock_calc.assert_not_called()

    def test_counts_failed_metric_calculation(
        self, mock_session_cls, mock_now, mock_calc, mock_update_run
    ) -> None:
        """Test that metric calculation errors are counted as failed."""
        session = mock_session_cls.return_value.__enter__.return_value
        mock_now.return_value = datetime(2026, 1, 1)

        stt_result = _make_result(id=1, stt_sample_id=100)
        sample = _make_sample(id=100, ground_truth="hello")

        session.exec.return_value.all.side_effect = [
            [stt_result],
            [sample],
        ]

        mock_calc.side_effect = ValueError("metric error")

        result = execute_metric_computation(**BASE_KWARGS)

        assert result["success"] is True
        assert result["scored"] == 0
        assert result["failed"] == 1
        # No bulk update since nothing scored
        session.execute.assert_not_called()
        mock_update_run.assert_not_called()

    def test_multiple_results_mixed_outcomes(
        self, mock_session_cls, mock_now, mock_calc, mock_update_run
    ) -> None:
        """Test mix of scored, skipped, and failed results."""
        session = mock_session_cls.return_value.__enter__.return_value
        mock_now.return_value = datetime(2026, 1, 1)

        # Result 1: will be scored
        result1 = _make_result(id=1, stt_sample_id=100, transcription="scored text")
        sample1 = _make_sample(id=100, ground_truth="scored text")

        # Result 2: will be skipped (no ground truth)
        result2 = _make_result(id=2, stt_sample_id=200, transcription="skipped text")
        sample2 = _make_sample(id=200, ground_truth=None)

        # Result 3: will fail (metric error)
        result3 = _make_result(id=3, stt_sample_id=300, transcription="failed text")
        sample3 = _make_sample(id=300, ground_truth="failed text")

        session.exec.return_value.all.side_effect = [
            [result1, result2, result3],  # results
            [sample1, sample2, sample3],  # samples
        ]

        scores = {"wer": 0.0, "cer": 0.0, "lenient_wer": 0.0, "wip": 1.0}
        mock_calc.side_effect = [scores, ValueError("error")]

        result = execute_metric_computation(**BASE_KWARGS)

        assert result["success"] is True
        assert result["scored"] == 1
        assert result["skipped"] == 1
        assert result["failed"] == 1

    def test_aggregate_scores_not_stored_when_nothing_scored(
        self, mock_session_cls, mock_now, mock_calc, mock_update_run
    ) -> None:
        """Test that run-level aggregate is not updated when no results were scored."""
        session = mock_session_cls.return_value.__enter__.return_value
        mock_now.return_value = datetime(2026, 1, 1)

        stt_result = _make_result(id=1, stt_sample_id=100)
        sample = _make_sample(id=100, ground_truth=None)  # will be skipped

        session.exec.return_value.all.side_effect = [
            [stt_result],
            [sample],
        ]

        execute_metric_computation(**BASE_KWARGS)

        mock_update_run.assert_not_called()
        session.execute.assert_not_called()

    def test_gevent_timeout_marks_run_failed_and_reraises(
        self, mock_session_cls, _mock_now, _mock_calc, mock_update_run
    ) -> None:
        """Test that a gevent Timeout marks the run as failed and re-raises."""
        session = mock_session_cls.return_value.__enter__.return_value
        session.exec.side_effect = Timeout()

        with pytest.raises(Timeout):
            execute_metric_computation(**BASE_KWARGS)

        mock_update_run.assert_called_once_with(
            session=session,
            run_id=BASE_KWARGS["run_id"],
            status="failed",
            error_message="Task exceeded soft time limit",
        )

    def test_soft_time_limit_exceeded_marks_run_failed_and_reraises(
        self, mock_session_cls, _mock_now, _mock_calc, mock_update_run
    ) -> None:
        """Test that SoftTimeLimitExceeded marks the run as failed and re-raises."""
        session = mock_session_cls.return_value.__enter__.return_value
        session.exec.side_effect = SoftTimeLimitExceeded()

        with pytest.raises(SoftTimeLimitExceeded):
            execute_metric_computation(**BASE_KWARGS)

        mock_update_run.assert_called_once_with(
            session=session,
            run_id=BASE_KWARGS["run_id"],
            status="failed",
            error_message="Task exceeded soft time limit",
        )
