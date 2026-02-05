import logging
from typing import Optional

from sqlmodel import Session, select

from app.models import Language

logger = logging.getLogger(__name__)


def get_languages(session: Session, skip: int = 0, limit: int = 100) -> list[Language]:
    """Retrieve all active languages."""
    statement = (
        select(Language).where(Language.is_active == True).offset(skip).limit(limit)
    )
    return list(session.exec(statement).all())


def get_language_by_id(session: Session, language_id: int) -> Optional[Language]:
    """Retrieve a language by its ID."""
    statement = select(Language).where(Language.id == language_id)
    return session.exec(statement).first()


def get_language_by_locale(session: Session, locale: str) -> Optional[Language]:
    """Retrieve a language by its locale code."""
    statement = select(Language).where(Language.locale == locale)
    return session.exec(statement).first()
