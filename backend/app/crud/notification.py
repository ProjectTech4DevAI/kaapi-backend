import logging
from typing import Any

from sqlmodel import Session, select

from app.core.util import now
from app.models import Notification, NotificationStatus

logger = logging.getLogger(__name__)


def notifications_exist_for_entity(
    *,
    session: Session,
    entity_type: str,
    entity_id: int,
    notification_type: str,
) -> bool:
    """Return True if any notification has already been recorded for this event.

    Used as the idempotency guard before creating a new fan-out of pending
    notifications for a terminal eval-run transition.
    """
    statement = select(Notification.id).where(
        Notification.entity_type == entity_type,
        Notification.entity_id == entity_id,
        Notification.notification_type == notification_type,
    )
    return session.exec(statement).first() is not None


def create_pending_notification(
    *,
    session: Session,
    notification_type: str,
    provider: str,
    recipient_user_id: int,
    entity_type: str,
    entity_id: int,
    project_id: int | None = None,
    subject: str | None = None,
    body_template: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Notification:
    """Insert a pending notification row. Caller commits."""
    notification = Notification(
        notification_type=notification_type,
        provider=provider,
        recipient_user_id=recipient_user_id,
        entity_type=entity_type,
        entity_id=entity_id,
        project_id=project_id,
        subject=subject,
        body_template=body_template,
        payload=payload or {},
        status=NotificationStatus.PENDING.value,
    )
    session.add(notification)
    return notification


def mark_notification_sent(*, session: Session, notification: Notification) -> None:
    notification.status = NotificationStatus.SENT.value
    notification.sent_at = now()
    notification.failed_reason = None
    notification.updated_at = now()
    session.add(notification)


def mark_notification_failed(
    *, session: Session, notification: Notification, reason: str
) -> None:
    notification.status = NotificationStatus.FAILED.value
    notification.failed_reason = reason[:2000] if reason else "Unknown error"
    notification.updated_at = now()
    session.add(notification)


def list_pending_notifications_for_entity(
    *,
    session: Session,
    entity_type: str,
    entity_id: int,
    notification_type: str,
) -> list[Notification]:
    statement = select(Notification).where(
        Notification.entity_type == entity_type,
        Notification.entity_id == entity_id,
        Notification.notification_type == notification_type,
        Notification.status == NotificationStatus.PENDING.value,
    )
    return list(session.exec(statement).all())
