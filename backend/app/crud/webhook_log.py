import logging
from uuid import UUID

from sqlmodel import Session, col, func, select

from app.models import WebhookDeliveryStatusEnum, WebhookLog, WebhookResourceTypeEnum

logger = logging.getLogger(__name__)


def create_webhook_log(*, session: Session, webhook_log: WebhookLog) -> WebhookLog:
    session.add(webhook_log)
    session.commit()
    logger.debug(
        f"[create_webhook_log] Webhook log created | id: {webhook_log.id} "
        f"| status: {webhook_log.status} | attempts: {webhook_log.attempt_count}"
    )
    return webhook_log


def get_webhook_log(
    *, session: Session, webhook_log_id: UUID, project_id: int, organization_id: int
) -> WebhookLog | None:
    statement = select(WebhookLog).where(
        WebhookLog.id == webhook_log_id,
        WebhookLog.project_id == project_id,
        WebhookLog.organization_id == organization_id,
    )
    return session.exec(statement).first()


def list_webhook_logs(
    *,
    session: Session,
    project_id: int,
    organization_id: int,
    resource_type: WebhookResourceTypeEnum | None = None,
    resource_id: str | None = None,
    status: WebhookDeliveryStatusEnum | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[WebhookLog], int]:
    filters = [
        WebhookLog.project_id == project_id,
        WebhookLog.organization_id == organization_id,
    ]
    if resource_type is not None:
        filters.append(WebhookLog.resource_type == resource_type)
    if resource_id is not None:
        filters.append(WebhookLog.resource_id == resource_id)
    if status is not None:
        filters.append(WebhookLog.status == status)

    total = session.exec(
        select(func.count()).select_from(WebhookLog).where(*filters)
    ).one()
    statement = (
        select(WebhookLog)
        .where(*filters)
        .order_by(col(WebhookLog.inserted_at).desc())
        .offset(skip)
        .limit(limit)
    )
    return list(session.exec(statement).all()), total
