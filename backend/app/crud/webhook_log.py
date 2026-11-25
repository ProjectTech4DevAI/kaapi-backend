from sqlmodel import Session

from app.models import WebhookLog


def create_webhook_log(*, session: Session, webhook_log: WebhookLog) -> None:
    session.add(webhook_log)
    session.commit()
