import logging
from sqlmodel import Session, select
from datetime import datetime
from app.models import OpenAIThreadCreate, OpenAIThread
from app.utils import mask_string

logger = logging.getLogger(__name__)


def upsert_thread_result(session: Session, data: OpenAIThreadCreate):
    statement = select(OpenAIThread).where(OpenAIThread.thread_id == data.thread_id)
    existing = session.exec(statement).first()

    if existing:
        existing.prompt = data.prompt
        existing.response = data.response
        existing.status = data.status
        existing.error = data.error
        existing.updated_at = datetime.utcnow()
        logger.info(
            f"[upsert_thread_result] Updated existing thread result in the db with ID: {mask_string(data.thread_id)}"
        )
    else:
        new_thread = OpenAIThread(**data.dict())
        session.add(new_thread)
        logger.info(
            f"[upsert_thread_result] Created new thread result in the db with ID: {mask_string(new_thread.thread_id)}"
        )
    session.commit()


def get_thread_result(session: Session, thread_id: str) -> OpenAIThread | None:
    statement = select(OpenAIThread).where(OpenAIThread.thread_id == thread_id)
    return session.exec(statement).first()
