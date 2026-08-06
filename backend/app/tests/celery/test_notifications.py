from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, select

from app.celery.tasks import job_execution as notif_task
from app.services.notifications.utils import content as notif_helpers
from app.crud.user_project import add_user_to_project
from app.services.notifications import eval_completion as eval_completion_service
from app.models import (
    EvaluationRun,
    Notification,
    NotificationEntityType,
    NotificationStatus,
    NotificationType,
    Project,
    User,
)
from app.tests.utils.test_data import (
    create_test_evaluation_dataset,
    create_test_project,
)
from app.tests.utils.utils import random_email


class _NonClosingSession:
    """Context-manager wrapper that yields a real session without closing it.

    The Celery task opens `Session(engine)` for its own scope; in tests we
    point that at the conftest's transactional session so changes roll back
    at the end of the test.
    """

    def __init__(self, session: Session):
        self._session = session

    def __enter__(self) -> Session:
        return self._session

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.fixture
def patched_session(db: Session):
    with patch(
        "app.services.notifications.eval_completion.Session",
        side_effect=lambda _engine: _NonClosingSession(db),
    ):
        yield db


def _emails_enabled(enabled: bool):
    """Toggle `settings.emails_enabled` by setting the underlying raw fields.

    `emails_enabled` is a Pydantic `@computed_field`, so `patch.object` can't
    override it on the instance — it's read-only. Instead, patch the two raw
    settings it derives from (`SMTP_HOST` + `EMAILS_FROM_EMAIL`) which is what
    `bool(SMTP_HOST and EMAILS_FROM_EMAIL)` evaluates against.
    """
    smtp_host = "smtp.test.local" if enabled else ""
    from_email = "noreply@test.local" if enabled else ""
    return _MultiPatch(
        patch.object(eval_completion_service.settings, "SMTP_HOST", smtp_host),
        patch.object(eval_completion_service.settings, "EMAILS_FROM_EMAIL", from_email),
    )


class _MultiPatch:
    """Tiny context manager that enters/exits multiple `patch` objects together."""

    def __init__(self, *patches):
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        for p in reversed(self._patches):
            p.stop()
        return False


def _add_active_member(db: Session, *, project: Project) -> str:
    """Add a fresh active user to the project and return their email.

    `add_user_to_project` creates new users with `is_active=False` (so they
    have to verify via invite before they can log in). The notification task
    filters those out — so for the fan-out tests we flip the flag immediately
    after creation to simulate an already-activated project member.
    """
    email = random_email()
    user, _ = add_user_to_project(
        session=db,
        email=email,
        organization_id=project.organization_id,
        project_id=project.id,
        full_name="Tester",
    )
    user.is_active = True
    db.add(user)
    db.commit()
    return email


def _make_eval_run(
    db: Session,
    *,
    project: Project,
    status: str = "completed",
    error_message: str | None = None,
    callback_url: str | None = None,
    score: dict | None = None,
    score_trace_url: str | None = None,
) -> EvaluationRun:
    dataset = create_test_evaluation_dataset(
        db,
        organization_id=project.organization_id,
        project_id=project.id,
    )
    run = EvaluationRun(
        run_name=f"test_run_{random_email()}",
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        type="text",
        status=status,
        organization_id=project.organization_id,
        project_id=project.id,
        updated_at=datetime(2026, 5, 16, 18, 33, 50),
        inserted_at=datetime(2026, 5, 16, 18, 0, 0),
        error_message=error_message,
        callback_url=callback_url,
        score=score,
        score_trace_url=score_trace_url,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


class TestHelperFunctions:
    def test_build_eval_results_link(self):
        eval_run = MagicMock(project_id=1, id=46)
        with patch.object(
            notif_helpers.settings, "FRONTEND_HOST", "http://example.com"
        ):
            assert (
                notif_helpers.build_eval_results_link(eval_run)
                == "http://example.com/evaluations/46"
            )

    def test_format_completed_at_converts_utc_to_ist(self):
        # 18:33 UTC + 5:30 = 00:03 IST next day
        assert (
            notif_helpers.format_completed_at(datetime(2026, 5, 16, 18, 33))
            == "May 17, 2026 at 12:03 AM"
        )

    def test_format_completed_at_strips_leading_zero(self):
        # 00:35 UTC + 5:30 = 06:05 IST
        assert (
            notif_helpers.format_completed_at(datetime(2026, 5, 16, 0, 35))
            == "May 16, 2026 at 6:05 AM"
        )

    def test_format_completed_at_handles_noon_and_midnight(self):
        # 06:30 UTC + 5:30 = 12:00 IST (noon)
        assert (
            notif_helpers.format_completed_at(datetime(2026, 5, 16, 6, 30))
            == "May 16, 2026 at 12:00 PM"
        )
        # 18:35 UTC + 5:30 = 00:05 IST (midnight next day)
        assert (
            notif_helpers.format_completed_at(datetime(2026, 5, 16, 18, 35))
            == "May 17, 2026 at 12:05 AM"
        )

    def test_format_completed_at_returns_empty_for_none(self):
        assert notif_helpers.format_completed_at(None) == ""

    def test_notification_type_for_status_failed(self):
        assert (
            notif_helpers.notification_type_for_status("failed")
            == NotificationType.EVAL_FAILED.value
        )

    def test_notification_type_for_status_completed(self):
        assert (
            notif_helpers.notification_type_for_status("completed")
            == NotificationType.EVAL_COMPLETED.value
        )

    def test_build_payload_includes_all_fields(self):
        eval_run = MagicMock(
            run_name="exp1",
            project_id=2,
            id=99,
            status="completed",
            updated_at=datetime(2026, 5, 16, 18, 33, 50),
            error_message=None,
        )
        with patch.object(
            notif_helpers.settings, "FRONTEND_HOST", "http://example.com"
        ):
            payload = notif_helpers.build_eval_completion_payload(
                eval_run=eval_run, project_name="ProjX"
            )
        # 18:33 UTC -> 00:03 IST next day
        assert payload == {
            "run_name": "exp1",
            "project_name": "ProjX",
            "status": "completed",
            "completed_at": "May 17, 2026 at 12:03 AM",
            "link": "http://example.com/evaluations/99",
            "error_message": None,
        }


class TestSendEvalCompletionNotification:
    def test_returns_not_found_when_eval_missing(self, patched_session: Session):
        result = notif_task.send_eval_completion_notification.apply(
            args=[99999999]
        ).result
        assert result == {
            "evaluation_id": 99999999,
            "sent": 0,
            "failed": 0,
            "not_found": True,
        }

    def test_skips_when_already_processed(self, patched_session: Session):
        project = create_test_project(patched_session)
        run = _make_eval_run(patched_session, project=project)

        recipient_user_id = patched_session.exec(select(User.id)).first()

        # Pre-existing notification row simulates a prior fan-out
        existing = Notification(
            notification_type=NotificationType.EVAL_COMPLETED.value,
            provider="email",
            recipient_user_id=recipient_user_id,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=run.id,
            project_id=project.id,
            payload={},
            status=NotificationStatus.SENT.value,
        )
        patched_session.add(existing)
        patched_session.commit()

        result = notif_task.send_eval_completion_notification.apply(
            args=[run.id]
        ).result
        assert result["skipped"] is True
        assert result["sent"] == 0

    def test_skips_when_emails_disabled(self, patched_session: Session):
        project = create_test_project(patched_session)
        run = _make_eval_run(patched_session, project=project)

        with _emails_enabled(False):
            result = notif_task.send_eval_completion_notification.apply(
                args=[run.id]
            ).result

        assert result["skipped"] is True
        # No notification rows created
        rows = patched_session.exec(
            select(Notification).where(Notification.entity_id == run.id)
        ).all()
        assert rows == []

    def test_no_recipients_returns_zero(self, patched_session: Session):
        project = create_test_project(patched_session)
        run = _make_eval_run(patched_session, project=project)

        with _emails_enabled(True):
            result = notif_task.send_eval_completion_notification.apply(
                args=[run.id]
            ).result

        assert result == {"evaluation_id": run.id, "sent": 0, "failed": 0}

    def test_fans_out_one_email_per_recipient(self, patched_session: Session):
        project = create_test_project(patched_session)
        run = _make_eval_run(patched_session, project=project)

        # Two activated project members
        _add_active_member(patched_session, project=project)
        _add_active_member(patched_session, project=project)

        with _emails_enabled(True), patch.object(
            eval_completion_service, "send_email"
        ) as mock_send:
            result = notif_task.send_eval_completion_notification.apply(
                args=[run.id]
            ).result

        assert result["recipients"] == 2
        assert result["sent"] == 2
        assert result["failed"] == 0
        assert mock_send.call_count == 2

        rows = patched_session.exec(
            select(Notification).where(Notification.entity_id == run.id)
        ).all()
        assert len(rows) == 2
        assert all(r.status == NotificationStatus.SENT.value for r in rows)
        assert all(r.sent_at is not None for r in rows)
        assert all(
            r.notification_type == NotificationType.EVAL_COMPLETED.value for r in rows
        )

    def test_marks_row_failed_when_smtp_raises(self, patched_session: Session):
        project = create_test_project(patched_session)
        run = _make_eval_run(patched_session, project=project)
        _add_active_member(patched_session, project=project)

        with _emails_enabled(True), patch.object(
            eval_completion_service,
            "send_email",
            side_effect=RuntimeError("SMTP timeout"),
        ):
            result = notif_task.send_eval_completion_notification.apply(
                args=[run.id]
            ).result

        assert result["sent"] == 0
        assert result["failed"] == 1

        rows = patched_session.exec(
            select(Notification).where(Notification.entity_id == run.id)
        ).all()
        assert len(rows) == 1
        assert rows[0].status == NotificationStatus.FAILED.value
        assert "SMTP timeout" in rows[0].failed_reason

    def test_failed_status_uses_eval_failed_type(self, patched_session: Session):
        project = create_test_project(patched_session)
        run = _make_eval_run(
            patched_session,
            project=project,
            status="failed",
            error_message="Batch failed",
        )
        _add_active_member(patched_session, project=project)

        with _emails_enabled(True), patch.object(eval_completion_service, "send_email"):
            result = notif_task.send_eval_completion_notification.apply(
                args=[run.id]
            ).result

        assert result["notification_type"] == NotificationType.EVAL_FAILED.value
        rows = patched_session.exec(
            select(Notification).where(Notification.entity_id == run.id)
        ).all()
        assert all(
            r.notification_type == NotificationType.EVAL_FAILED.value for r in rows
        )


class TestExecuteEvalCompletionCallback:
    """Best-effort webhook delivery for a terminal v2 run.

    `send_callback` and `get_webhook_secret` are the external boundaries (HTTP +
    credential lookup), patched at the service's import site. The per-row trace
    load (`_load_cached_traces`, which hits S3) is also mocked; the run row itself
    is real, so the APIResponse envelope is built from a genuine EvaluationRunPublic.
    """

    _HOOK = "https://hooks.example.com/eval"
    _SUMMARY_SCORE = {
        "summary_scores": [{"name": "cosine", "value": 0.9}],
        "overall": 0.9,
    }
    _S3_URL = "https://s3.example.com/traces/eval.json"
    _TRACES = [{"trace_id": "t1", "score": 0.8}, {"trace_id": "t2", "score": 1.0}]

    def test_completed_run_with_trace_url_ships_traces_and_no_leak(
        self, patched_session: Session
    ):
        """A finalized completed run attaches S3 traces onto data.score, matching GET."""
        project = create_test_project(patched_session)
        run = _make_eval_run(
            patched_session,
            project=project,
            status="completed",
            callback_url=self._HOOK,
            score=self._SUMMARY_SCORE,
            score_trace_url=self._S3_URL,
        )

        with (
            patch.object(
                eval_completion_service,
                "_load_cached_traces",
                return_value=(self._TRACES, False),
            ),
            patch.object(
                eval_completion_service, "get_webhook_secret", return_value="s3cr3t"
            ),
            patch.object(
                eval_completion_service, "send_callback", return_value=True
            ) as mock_send,
        ):
            result = eval_completion_service.execute_eval_completion_callback(run.id)

        assert result == {"evaluation_id": run.id, "delivered": True}
        mock_send.assert_called_once()
        args, kwargs = mock_send.call_args
        assert args[0] == self._HOOK
        assert kwargs["webhook_secret"] == "s3cr3t"

        payload = args[1]
        assert payload["success"] is True
        data = payload["data"]
        assert data["id"] == run.id
        assert data["status"] == "completed"
        assert data["score_trace_url"] == self._S3_URL
        assert data["score"]["traces"] == self._TRACES
        assert data["score"]["summary_scores"] == self._SUMMARY_SCORE["summary_scores"]
        assert data["score"]["overall"] == self._SUMMARY_SCORE["overall"]
        assert "callback_url" not in data

    def test_completed_run_without_trace_url_requests_retry(
        self, patched_session: Session
    ):
        """Fired before S3 finalized: signal retry, do not deliver."""
        project = create_test_project(patched_session)
        run = _make_eval_run(
            patched_session,
            project=project,
            status="completed",
            callback_url=self._HOOK,
            score=self._SUMMARY_SCORE,
            score_trace_url=None,
        )

        with patch.object(eval_completion_service, "send_callback") as mock_send:
            result = eval_completion_service.execute_eval_completion_callback(run.id)

        mock_send.assert_not_called()
        assert result == {
            "evaluation_id": run.id,
            "delivered": False,
            "retry": True,
        }

    def test_force_send_completed_without_trace_url_delivers_without_traces(
        self, patched_session: Session
    ):
        """The on-exhaustion fallback delivers the summary-only payload anyway."""
        project = create_test_project(patched_session)
        run = _make_eval_run(
            patched_session,
            project=project,
            status="completed",
            callback_url=self._HOOK,
            score=self._SUMMARY_SCORE,
            score_trace_url=None,
        )

        with (
            patch.object(
                eval_completion_service, "get_webhook_secret", return_value=None
            ),
            patch.object(
                eval_completion_service, "send_callback", return_value=True
            ) as mock_send,
        ):
            result = eval_completion_service.execute_eval_completion_callback(
                run.id, force_send=True
            )

        assert result == {"evaluation_id": run.id, "delivered": True}
        payload = mock_send.call_args.args[1]
        assert "traces" not in payload["data"]["score"]
        assert payload["data"]["score_trace_url"] is None

    def test_failed_run_delivers_immediately_without_traces(
        self, patched_session: Session
    ):
        """Failed runs never have a trace URL: no retry, failure envelope, no traces."""
        project = create_test_project(patched_session)
        run = _make_eval_run(
            patched_session,
            project=project,
            status="failed",
            error_message="judge crashed",
            callback_url=self._HOOK,
        )

        with (
            patch.object(
                eval_completion_service, "get_webhook_secret", return_value=None
            ),
            patch.object(
                eval_completion_service, "send_callback", return_value=True
            ) as mock_send,
        ):
            result = eval_completion_service.execute_eval_completion_callback(run.id)

        assert result == {"evaluation_id": run.id, "delivered": True}
        payload = mock_send.call_args.args[1]
        assert payload["success"] is False
        assert payload["error"] == "judge crashed"
        assert payload["data"]["id"] == run.id
        assert payload["data"]["status"] == "failed"
        assert payload["data"]["score"] is None

    def test_run_without_callback_url_skips_delivery(self, patched_session: Session):
        project = create_test_project(patched_session)
        run = _make_eval_run(
            patched_session, project=project, status="completed", callback_url=None
        )

        with patch.object(eval_completion_service, "send_callback") as mock_send:
            result = eval_completion_service.execute_eval_completion_callback(run.id)

        mock_send.assert_not_called()
        assert result == {
            "evaluation_id": run.id,
            "delivered": False,
            "skipped": True,
        }

    def test_delivery_exception_is_swallowed(self, patched_session: Session):
        project = create_test_project(patched_session)
        run = _make_eval_run(
            patched_session,
            project=project,
            status="completed",
            callback_url=self._HOOK,
            score=self._SUMMARY_SCORE,
            score_trace_url=self._S3_URL,
        )

        with (
            patch.object(
                eval_completion_service,
                "_load_cached_traces",
                return_value=(self._TRACES, False),
            ),
            patch.object(
                eval_completion_service, "get_webhook_secret", return_value=None
            ),
            patch.object(
                eval_completion_service,
                "send_callback",
                side_effect=RuntimeError("connection reset"),
            ),
        ):
            result = eval_completion_service.execute_eval_completion_callback(run.id)

        assert result == {"evaluation_id": run.id, "delivered": False}


class TestSendEvalCompletionCallbackTaskRetry:
    """Retry wiring in the Celery task shim around execute_eval_completion_callback."""

    def test_retry_signal_triggers_self_retry(self):
        with (
            patch.object(
                eval_completion_service,
                "execute_eval_completion_callback",
                return_value={"evaluation_id": 5, "delivered": False, "retry": True},
            ),
            patch.object(
                notif_task.send_eval_completion_callback,
                "retry",
                side_effect=RuntimeError("retry-raised"),
            ) as mock_retry,
        ):
            with pytest.raises(RuntimeError, match="retry-raised"):
                notif_task.send_eval_completion_callback.apply(args=[5]).get()

        mock_retry.assert_called_once()
        assert mock_retry.call_args.kwargs["countdown"] == 5
        assert mock_retry.call_args.kwargs["max_retries"] == 5

    def test_max_retries_exceeded_falls_back_to_force_send(self):
        from celery.exceptions import MaxRetriesExceededError

        calls: list[dict] = []

        def _fake_execute(evaluation_id: int, force_send: bool = False) -> dict:
            calls.append({"evaluation_id": evaluation_id, "force_send": force_send})
            if not force_send:
                return {
                    "evaluation_id": evaluation_id,
                    "delivered": False,
                    "retry": True,
                }
            return {"evaluation_id": evaluation_id, "delivered": True}

        with (
            patch.object(
                eval_completion_service,
                "execute_eval_completion_callback",
                side_effect=_fake_execute,
            ),
            patch.object(
                notif_task.send_eval_completion_callback,
                "retry",
                side_effect=MaxRetriesExceededError(),
            ),
        ):
            result = notif_task.send_eval_completion_callback.apply(args=[9]).get()

        assert result == {"evaluation_id": 9, "delivered": True}
        assert calls == [
            {"evaluation_id": 9, "force_send": False},
            {"evaluation_id": 9, "force_send": True},
        ]

    def test_no_retry_signal_returns_result_directly(self):
        with (
            patch.object(
                eval_completion_service,
                "execute_eval_completion_callback",
                return_value={"evaluation_id": 3, "delivered": True},
            ),
            patch.object(
                notif_task.send_eval_completion_callback, "retry"
            ) as mock_retry,
        ):
            result = notif_task.send_eval_completion_callback.apply(args=[3]).get()

        mock_retry.assert_not_called()
        assert result == {"evaluation_id": 3, "delivered": True}
