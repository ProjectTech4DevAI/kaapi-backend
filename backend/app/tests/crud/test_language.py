from sqlmodel import Session

from app.crud.language import get_language_by_locale, get_language_locale_by_id
from app.models.language import Language
from app.tests.utils.utils import get_non_existent_id


def test_get_language_locale_by_id_returns_seeded_locale(db: Session) -> None:
    hindi = get_language_by_locale(session=db, locale="hi")

    assert get_language_locale_by_id(session=db, language_id=hindi.id) == "hi"


def test_get_language_locale_by_id_returns_none_when_missing(db: Session) -> None:
    missing_id = get_non_existent_id(db, Language)

    assert get_language_locale_by_id(session=db, language_id=missing_id) is None
