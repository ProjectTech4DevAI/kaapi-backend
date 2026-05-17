from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, select

from app.celery.tasks import notifications as notif_task
from app.crud.user_project import add_user_to_project
from app.models import (
    EvaluationRun,
    Notification,
    NotificationEntityType,
    NotificationStatus,
    NotificationType,
    Project,
)
from app.tests.utils.test_data import create_test_project
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
        "app.celery.tasks.notifications.Session",
        side_effect=lambda _engine: _NonClosingSession(db),
    ):
        yield db


def _make_eval_run(
    db: Session,
    *,
    project: Project,
    status: str = "completed",
    error_message: str | None = None,
) -> EvaluationRun:
    run = EvaluationRun(
        run_name=f"test_run_{random_email()}",
        dataset_name="dataset_x",
        dataset_id=1,
        type="text",
        status=status,
        organization_id=project.organization_id,
        project_id=project.id,
        updated_at=datetime(2026, 5, 16, 18, 33, 50),
        inserted_at=datetime(2026, 5, 16, 18, 0, 0),
        error_message=error_message,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


class TestHelperFunctions:
    def test_build_eval_results_link(self):
        eval_run = MagicMock(project_id=1, id=46)
        with patch.object(notif_task.settings, "FRONTEND_HOST", "http://example.com"):
            assert (
                notif_task._build_eval_results_link(eval_run)
                == "http://example.com/evaluations/"
            )

    def test_notification_type_for_status_failed(self):
        assert (
            notif_task._notification_type_for_status("failed")
            == NotificationType.EVAL_FAILED.value
        )

    def test_notification_type_for_status_completed(self):
        assert (
            notif_task._notification_type_for_status("completed")
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
        with patch.object(notif_task.settings, "FRONTEND_HOST", "http://example.com"):
            payload = notif_task._build_eval_completion_payload(
                eval_run=eval_run, project_name="ProjX"
            )
        assert payload == {
            "run_name": "exp1",
            "project_name": "ProjX",
            "status": "completed",
            "completed_at": "2026-05-16T18:33:50",
            "link": "http://example.com/evaluations/",
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

        # Pre-existing notification row simulates a prior fan-out
        existing = Notification(
            notification_type=NotificationType.EVAL_COMPLETED.value,
            provider="email",
            recipient_user_id=1,
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

        with patch.object(notif_task.settings, "emails_enabled", False):
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

        with patch.object(notif_task.settings, "emails_enabled", True):
            result = notif_task.send_eval_completion_notification.apply(
                args=[run.id]
            ).result

        assert result == {"evaluation_id": run.id, "sent": 0, "failed": 0}

    def test_fans_out_one_email_per_recipient(self, patched_session: Session):
        project = create_test_project(patched_session)
        run = _make_eval_run(patched_session, project=project)

        # Add two project members
        for _ in range(2):
            add_user_to_project(
                session=patched_session,
                email=random_email(),
                organization_id=project.organization_id,
                project_id=project.id,
                full_name="Tester",
            )
        patched_session.commit()

        with patch.object(notif_task.settings, "emails_enabled", True), patch.object(
            notif_task, "send_email"
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
        add_user_to_project(
            session=patched_session,
            email=random_email(),
            organization_id=project.organization_id,
            project_id=project.id,
        )
        patched_session.commit()

        with patch.object(notif_task.settings, "emails_enabled", True), patch.object(
            notif_task, "send_email", side_effect=RuntimeError("SMTP timeout")
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
        add_user_to_project(
            session=patched_session,
            email=random_email(),
            organization_id=project.organization_id,
            project_id=project.id,
        )
        patched_session.commit()

        with patch.object(notif_task.settings, "emails_enabled", True), patch.object(
            notif_task, "send_email"
        ):
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
