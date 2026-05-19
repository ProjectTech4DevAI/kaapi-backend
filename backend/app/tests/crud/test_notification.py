from sqlmodel import Session

from app.crud.notification import (
    create_pending_notification,
    list_pending_notifications_for_entity,
    mark_notification_failed,
    mark_notification_sent,
    notifications_exist_for_entity,
)
from app.models import (
    NotificationEntityType,
    NotificationProvider,
    NotificationStatus,
    NotificationType,
)
from app.tests.utils.test_data import create_test_project
from app.tests.utils.user import create_random_user


def _make_pending(
    db: Session,
    *,
    project_id: int | None,
    recipient_user_id: int,
    entity_id: int,
    notification_type: str = NotificationType.EVAL_COMPLETED.value,
    entity_type: str = NotificationEntityType.EVAL_RUN.value,
    payload: dict | None = None,
):
    return create_pending_notification(
        session=db,
        notification_type=notification_type,
        provider=NotificationProvider.EMAIL.value,
        recipient_user_id=recipient_user_id,
        entity_type=entity_type,
        entity_id=entity_id,
        project_id=project_id,
        subject="Test subject",
        body_template="eval_completion_v1",
        payload=payload or {"foo": "bar"},
    )


class TestNotificationCRUD:
    def test_create_pending_persists_with_pending_status(self, db: Session):
        project = create_test_project(db)
        user = create_random_user(db)

        notif = _make_pending(
            db,
            project_id=project.id,
            recipient_user_id=user.id,
            entity_id=42,
        )
        db.commit()
        db.refresh(notif)

        assert notif.id is not None
        assert notif.status == NotificationStatus.PENDING.value
        assert notif.sent_at is None
        assert notif.failed_reason is None
        assert notif.notification_type == "eval_completed"
        assert notif.provider == "email"
        assert notif.project_id == project.id
        assert notif.payload == {"foo": "bar"}

    def test_create_pending_allows_null_project(self, db: Session):
        """Magic-link notifications can have project_id=None."""
        user = create_random_user(db)

        notif = _make_pending(
            db,
            project_id=None,
            recipient_user_id=user.id,
            entity_id=user.id,
            notification_type=NotificationType.MAGIC_LINK_LOGIN.value,
            entity_type=NotificationEntityType.USER.value,
        )
        db.commit()
        db.refresh(notif)

        assert notif.project_id is None

    def test_create_pending_defaults_payload_to_empty(self, db: Session):
        user = create_random_user(db)
        project = create_test_project(db)

        notif = create_pending_notification(
            session=db,
            notification_type=NotificationType.MAGIC_LINK_LOGIN.value,
            provider=NotificationProvider.EMAIL.value,
            recipient_user_id=user.id,
            entity_type=NotificationEntityType.USER.value,
            entity_id=user.id,
            project_id=project.id,
        )
        db.commit()
        db.refresh(notif)

        assert notif.payload == {}
        assert notif.subject is None
        assert notif.body_template is None

    def test_notifications_exist_for_entity(self, db: Session):
        project = create_test_project(db)
        user = create_random_user(db)

        assert not notifications_exist_for_entity(
            session=db,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=99,
            notification_type=NotificationType.EVAL_COMPLETED.value,
        )

        _make_pending(
            db,
            project_id=project.id,
            recipient_user_id=user.id,
            entity_id=99,
        )
        db.commit()

        assert notifications_exist_for_entity(
            session=db,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=99,
            notification_type=NotificationType.EVAL_COMPLETED.value,
        )

    def test_notifications_exist_is_scoped_by_type(self, db: Session):
        """Existence check is keyed on (entity_type, entity_id, notification_type)."""
        project = create_test_project(db)
        user = create_random_user(db)

        _make_pending(
            db,
            project_id=project.id,
            recipient_user_id=user.id,
            entity_id=7,
            notification_type=NotificationType.EVAL_COMPLETED.value,
        )
        db.commit()

        assert notifications_exist_for_entity(
            session=db,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=7,
            notification_type=NotificationType.EVAL_COMPLETED.value,
        )
        # Different notification_type → no match
        assert not notifications_exist_for_entity(
            session=db,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=7,
            notification_type=NotificationType.EVAL_FAILED.value,
        )

    def test_mark_notification_sent(self, db: Session):
        project = create_test_project(db)
        user = create_random_user(db)
        notif = _make_pending(
            db, project_id=project.id, recipient_user_id=user.id, entity_id=11
        )
        db.commit()
        db.refresh(notif)

        mark_notification_sent(session=db, notification=notif)
        db.commit()
        db.refresh(notif)

        assert notif.status == NotificationStatus.SENT.value
        assert notif.sent_at is not None
        assert notif.failed_reason is None

    def test_mark_notification_failed(self, db: Session):
        project = create_test_project(db)
        user = create_random_user(db)
        notif = _make_pending(
            db, project_id=project.id, recipient_user_id=user.id, entity_id=12
        )
        db.commit()
        db.refresh(notif)

        mark_notification_failed(
            session=db, notification=notif, reason="SMTP refused connection"
        )
        db.commit()
        db.refresh(notif)

        assert notif.status == NotificationStatus.FAILED.value
        assert notif.failed_reason == "SMTP refused connection"

    def test_mark_notification_failed_truncates_long_reason(self, db: Session):
        """Long failure messages are trimmed to 2000 chars to fit the column."""
        project = create_test_project(db)
        user = create_random_user(db)
        notif = _make_pending(
            db, project_id=project.id, recipient_user_id=user.id, entity_id=13
        )
        db.commit()
        db.refresh(notif)

        long_reason = "x" * 5000
        mark_notification_failed(session=db, notification=notif, reason=long_reason)
        db.commit()
        db.refresh(notif)

        assert len(notif.failed_reason) == 2000

    def test_mark_notification_failed_handles_empty_reason(self, db: Session):
        project = create_test_project(db)
        user = create_random_user(db)
        notif = _make_pending(
            db, project_id=project.id, recipient_user_id=user.id, entity_id=14
        )
        db.commit()
        db.refresh(notif)

        mark_notification_failed(session=db, notification=notif, reason="")
        db.commit()
        db.refresh(notif)

        assert notif.failed_reason == "Unknown error"

    def test_list_pending_notifications_for_entity_filters_status(self, db: Session):
        project = create_test_project(db)
        u1 = create_random_user(db)
        u2 = create_random_user(db)

        n1 = _make_pending(
            db, project_id=project.id, recipient_user_id=u1.id, entity_id=21
        )
        n2 = _make_pending(
            db, project_id=project.id, recipient_user_id=u2.id, entity_id=21
        )
        db.commit()

        # Both pending initially
        pending = list_pending_notifications_for_entity(
            session=db,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=21,
            notification_type=NotificationType.EVAL_COMPLETED.value,
        )
        assert {n.id for n in pending} == {n1.id, n2.id}

        # Mark one sent; only the still-pending row should be returned
        mark_notification_sent(session=db, notification=n1)
        db.commit()

        pending = list_pending_notifications_for_entity(
            session=db,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=21,
            notification_type=NotificationType.EVAL_COMPLETED.value,
        )
        assert [n.id for n in pending] == [n2.id]
