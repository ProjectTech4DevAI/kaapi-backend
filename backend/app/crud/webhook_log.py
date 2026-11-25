import logging

from sqlmodel import Session

from app.models import WebhookLog

logger = logging.getLogger(__name__)


def create_webhook_log(*, session: Session, webhook_log: WebhookLog) -> WebhookLog:
    session.add(webhook_log)
    session.commit()
    logger.debug(
        f"[create_webhook_log] Webhook log created | id: {webhook_log.id} "
        f"| status: {webhook_log.status} | attempts: {webhook_log.attempt_count}"
    )
    return webhook_log
