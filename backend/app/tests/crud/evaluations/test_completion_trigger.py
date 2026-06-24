"""Tests for the completion-notification trigger inside update_evaluation_run.

These verify that:
  - terminal transitions (processing -> completed/failed) enqueue the task
  - non-transitions (status unchanged, score-only updates) do not
  - broker errors during enqueue do not bubble up
"""

from unittest.mock import MagicMock, patch

from app.crud.evaluations.core import (
    TERMINAL_EVAL_STATUSES,
    update_evaluation_run,
)
from app.models import EvaluationRun, EvaluationRunUpdate


def _stub_eval_run(status: str = "processing") -> MagicMock:
    eval_run = MagicMock(spec=EvaluationRun)
    eval_run.id = 7
    eval_run.status = status
    return eval_run


class TestTerminalStateTrigger:
    def test_terminal_set_constants(self):
        assert TERMINAL_EVAL_STATUSES == {"completed", "failed"}

    def test_transition_to_completed_enqueues(self):
        eval_run = _stub_eval_run(status="processing")
        session = MagicMock()

        with patch(
            "app.crud.evaluations.core._enqueue_eval_completion_notification"
        ) as enq:
            update_evaluation_run(
                session=session,
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="completed"),
            )
            enq.assert_called_once_with(eval_run)

    def test_transition_to_failed_enqueues(self):
        eval_run = _stub_eval_run(status="processing")
        session = MagicMock()

        with patch(
            "app.crud.evaluations.core._enqueue_eval_completion_notification"
        ) as enq:
            update_evaluation_run(
                session=session,
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="failed", error_message="boom"),
            )
            enq.assert_called_once_with(eval_run)

    def test_re_set_completed_does_not_re_enqueue(self):
        """A second update with status=completed on an already-completed row is a no-op."""
        eval_run = _stub_eval_run(status="completed")
        session = MagicMock()

        with patch(
            "app.crud.evaluations.core._enqueue_eval_completion_notification"
        ) as enq:
            update_evaluation_run(
                session=session,
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="completed"),
            )
            enq.assert_not_called()

    def test_score_only_update_does_not_enqueue(self):
        """Updates without a status field never trigger a notification."""
        eval_run = _stub_eval_run(status="completed")
        session = MagicMock()

        with patch(
            "app.crud.evaluations.core._enqueue_eval_completion_notification"
        ) as enq:
            update_evaluation_run(
                session=session,
                eval_run=eval_run,
                update=EvaluationRunUpdate(score={"foo": 1}),
            )
            enq.assert_not_called()

    def test_processing_status_does_not_enqueue(self):
        """Transition into non-terminal state (e.g. processing) does not notify."""
        eval_run = _stub_eval_run(status="pending")
        session = MagicMock()

        with patch(
            "app.crud.evaluations.core._enqueue_eval_completion_notification"
        ) as enq:
            update_evaluation_run(
                session=session,
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="processing"),
            )
            enq.assert_not_called()

    def test_enqueue_broker_failure_does_not_raise(self):
        """If celery .delay() throws, the update still succeeds."""
        eval_run = _stub_eval_run(status="processing")
        session = MagicMock()

        with patch(
            "app.celery.tasks.job_execution.send_eval_completion_notification.delay",
            side_effect=RuntimeError("broker down"),
        ):
            # Should NOT raise — the helper swallows broker errors and logs them.
            update_evaluation_run(
                session=session,
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="completed"),
            )
