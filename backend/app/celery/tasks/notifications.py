import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlmodel import Session

from app.celery.celery_app import celery_app
from app.celery.utils import gevent_timeout
from app.core.config import settings
from app.core.db import engine
from app.crud.notification import (
    create_pending_notification,
    list_pending_notifications_for_entity,
    mark_notification_failed,
    mark_notification_sent,
    notifications_exist_for_entity,
)
from app.crud.user_project import get_users_by_project
from app.models import (
    EvaluationRun,
    NotificationEntityType,
    NotificationProvider,
    NotificationType,
    Project,
)
from app.utils import generate_eval_completion_email, send_email

logger = logging.getLogger(__name__)

EVAL_COMPLETION_TEMPLATE = "eval_completion_v1"
IST_TZ = ZoneInfo("Asia/Kolkata")


def _build_eval_results_link(eval_run: EvaluationRun) -> str:
    return f"{settings.FRONTEND_HOST}/evaluations/{eval_run.id}"


def _notification_type_for_status(status: str) -> str:
    if status == "failed":
        return NotificationType.EVAL_FAILED.value
    return NotificationType.EVAL_COMPLETED.value


def _format_completed_at(dt: datetime | None) -> str:
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(IST_TZ)
    hour_12 = local.strftime("%I").lstrip("0") or "12"
    return local.strftime(f"%B %d, %Y at {hour_12}:%M %p")


def _build_eval_completion_payload(
    *, eval_run: EvaluationRun, project_name: str
) -> dict:
    return {
        "run_name": eval_run.run_name,
        "project_name": project_name,
        "status": eval_run.status,
        "completed_at": _format_completed_at(eval_run.updated_at),
        "link": _build_eval_results_link(eval_run),
        "error_message": eval_run.error_message,
    }


@celery_app.task(bind=True, queue="default", priority=1)
@gevent_timeout(
    settings.CELERY_TASK_SOFT_TIME_LIMIT, "send_eval_completion_notification"
)
def send_eval_completion_notification(self, evaluation_id: int) -> dict:
    """
    Fan out a completion notification for an eval run to every project member.

    The flow per recipient is: insert a `pending` notification row with the
    payload snapshot, attempt SMTP delivery, then flip the row to `sent`
    (with `sent_at`) or `failed` (with `failed_reason`). The `notification`
    table itself acts as the idempotency guard — if any rows already exist
    for this (entity_type, entity_id, notification_type), the task bails
    out without sending again.
    """
    with Session(engine) as session:
        eval_run = session.get(EvaluationRun, evaluation_id)
        if not eval_run:
            logger.error(
                f"[send_eval_completion_notification] EvaluationRun not found | "
                f"evaluation_id={evaluation_id}"
            )
            return {
                "evaluation_id": evaluation_id,
                "sent": 0,
                "failed": 0,
                "not_found": True,
            }

        notification_type = _notification_type_for_status(eval_run.status)

        already_processed = notifications_exist_for_entity(
            session=session,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=eval_run.id,
            notification_type=notification_type,
        )
        if already_processed:
            logger.info(
                f"[send_eval_completion_notification] Already processed; skipping | "
                f"evaluation_id={evaluation_id} | type={notification_type}"
            )
            return {
                "evaluation_id": evaluation_id,
                "sent": 0,
                "failed": 0,
                "skipped": True,
            }

        if not settings.emails_enabled:
            logger.warning(
                f"[send_eval_completion_notification] Email not configured; skipping | "
                f"evaluation_id={evaluation_id}"
            )
            return {
                "evaluation_id": evaluation_id,
                "sent": 0,
                "failed": 0,
                "skipped": True,
            }

        project = session.get(Project, eval_run.project_id)
        project_name = project.name if project else f"Project {eval_run.project_id}"

        users = get_users_by_project(session=session, project_id=eval_run.project_id)
        recipients = [u for u in users if u.is_active and u.email]
        if not recipients:
            logger.info(
                f"[send_eval_completion_notification] No recipients for project | "
                f"evaluation_id={evaluation_id} | project_id={eval_run.project_id}"
            )
            return {"evaluation_id": evaluation_id, "sent": 0, "failed": 0}

        payload = _build_eval_completion_payload(
            eval_run=eval_run, project_name=project_name
        )
        email_data = generate_eval_completion_email(
            run_name=eval_run.run_name,
            project_name=project_name,
            status=eval_run.status,
            completed_at=payload["completed_at"],
            link=payload["link"],
            error_message=eval_run.error_message,
        )

        seen_user_ids: set[int] = set()
        for user in recipients:
            if user.user_id in seen_user_ids:
                continue
            seen_user_ids.add(user.user_id)
            create_pending_notification(
                session=session,
                notification_type=notification_type,
                provider=NotificationProvider.EMAIL.value,
                recipient_user_id=user.user_id,
                entity_type=NotificationEntityType.EVAL_RUN.value,
                entity_id=eval_run.id,
                project_id=eval_run.project_id,
                subject=email_data.subject,
                body_template=EVAL_COMPLETION_TEMPLATE,
                payload=payload,
            )
        session.commit()

        pending = list_pending_notifications_for_entity(
            session=session,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=eval_run.id,
            notification_type=notification_type,
        )

        sent_count = 0
        failed_count = 0
        for notification in pending:
            email_to = next(
                (
                    u.email
                    for u in recipients
                    if u.user_id == notification.recipient_user_id
                ),
                None,
            )
            if not email_to:
                mark_notification_failed(
                    session=session,
                    notification=notification,
                    reason="Recipient email not available",
                )
                failed_count += 1
                continue
            try:
                send_email(
                    email_to=email_to,
                    subject=email_data.subject,
                    html_content=email_data.html_content,
                )
                mark_notification_sent(session=session, notification=notification)
                sent_count += 1
                logger.info(
                    f"[send_eval_completion_notification] Sent | "
                    f"evaluation_id={evaluation_id} | "
                    f"notification_id={notification.id} | to={email_to}"
                )
            except Exception as e:
                mark_notification_failed(
                    session=session, notification=notification, reason=str(e)
                )
                failed_count += 1
                logger.error(
                    f"[send_eval_completion_notification] Send failed | "
                    f"evaluation_id={evaluation_id} | "
                    f"notification_id={notification.id} | to={email_to} | error={e}",
                    exc_info=True,
                )
        session.commit()

        logger.info(
            f"[send_eval_completion_notification] Done | "
            f"evaluation_id={evaluation_id} | project_id={eval_run.project_id} | "
            f"type={notification_type} | recipients={len(pending)} | "
            f"sent={sent_count} | failed={failed_count}"
        )
        return {
            "evaluation_id": evaluation_id,
            "notification_type": notification_type,
            "recipients": len(pending),
            "sent": sent_count,
            "failed": failed_count,
        }
