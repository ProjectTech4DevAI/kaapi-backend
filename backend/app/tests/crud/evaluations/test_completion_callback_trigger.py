"""Tests for the completion-callback (webhook) trigger inside update_evaluation_run.

Sibling to test_completion_trigger.py, which covers the email fan-out. Here we
verify the v2 callback hook: a run that registered a `callback_url` enqueues the
webhook task exactly once on the first terminal transition (completed or failed),
never fires without a URL or on a non-transition, and swallows broker errors.
The email enqueue is patched out so the two paths stay independent.
"""

from unittest.mock import MagicMock, patch

from app.crud.evaluations.core import update_evaluation_run
from app.models import EvaluationRun, EvaluationRunUpdate


def _stub_eval_run(
    *,
    status: str = "processing",
    callback_url: str | None = "https://hooks.example.com/eval",
) -> MagicMock:
    eval_run = MagicMock(spec=EvaluationRun)
    eval_run.id = 7
    eval_run.status = status
    eval_run.callback_url = callback_url
    return eval_run


class TestCompletionCallbackTrigger:
    def test_transition_to_completed_enqueues_callback(self):
        eval_run = _stub_eval_run(status="processing")
        with (
            patch("app.crud.evaluations.core._enqueue_eval_completion_notification"),
            patch(
                "app.celery.tasks.job_execution.send_eval_completion_callback.delay"
            ) as delay,
        ):
            update_evaluation_run(
                session=MagicMock(),
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="completed"),
            )
        delay.assert_called_once_with(eval_run.id)

    def test_transition_to_failed_enqueues_callback(self):
        eval_run = _stub_eval_run(status="processing")
        with (
            patch("app.crud.evaluations.core._enqueue_eval_completion_notification"),
            patch(
                "app.celery.tasks.job_execution.send_eval_completion_callback.delay"
            ) as delay,
        ):
            update_evaluation_run(
                session=MagicMock(),
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="failed", error_message="boom"),
            )
        delay.assert_called_once_with(eval_run.id)

    def test_no_callback_url_does_not_enqueue_callback(self):
        """The email path still fires; only the webhook is skipped without a URL."""
        eval_run = _stub_eval_run(status="processing", callback_url=None)
        with (
            patch(
                "app.crud.evaluations.core._enqueue_eval_completion_notification"
            ) as email_enq,
            patch(
                "app.celery.tasks.job_execution.send_eval_completion_callback.delay"
            ) as delay,
        ):
            update_evaluation_run(
                session=MagicMock(),
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="completed"),
            )
        delay.assert_not_called()
        email_enq.assert_called_once_with(eval_run)

    def test_terminal_to_terminal_does_not_enqueue_callback(self):
        eval_run = _stub_eval_run(status="completed")
        with (
            patch("app.crud.evaluations.core._enqueue_eval_completion_notification"),
            patch(
                "app.celery.tasks.job_execution.send_eval_completion_callback.delay"
            ) as delay,
        ):
            update_evaluation_run(
                session=MagicMock(),
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="failed", error_message="late"),
            )
        delay.assert_not_called()

    def test_non_terminal_transition_does_not_enqueue_callback(self):
        eval_run = _stub_eval_run(status="pending")
        with (
            patch("app.crud.evaluations.core._enqueue_eval_completion_notification"),
            patch(
                "app.celery.tasks.job_execution.send_eval_completion_callback.delay"
            ) as delay,
        ):
            update_evaluation_run(
                session=MagicMock(),
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="processing"),
            )
        delay.assert_not_called()

    def test_broker_failure_does_not_raise(self):
        eval_run = _stub_eval_run(status="processing")
        with (
            patch("app.crud.evaluations.core._enqueue_eval_completion_notification"),
            patch(
                "app.celery.tasks.job_execution.send_eval_completion_callback.delay",
                side_effect=RuntimeError("broker down"),
            ),
        ):
            update_evaluation_run(
                session=MagicMock(),
                eval_run=eval_run,
                update=EvaluationRunUpdate(status="completed"),
            )
