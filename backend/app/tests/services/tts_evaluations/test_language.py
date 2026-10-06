import logging

import pytest
from sqlmodel import Session

from app.crud.language import get_language_by_locale
from app.models.language import Language
from app.services.tts_evaluations.language import (
    resolve_dataset_tts_language_code,
    resolve_tts_language_code,
    to_elevenlabs_language_code,
)
from app.tests.utils.utils import get_non_existent_id


class TestResolveTTSLanguageCode:
    @pytest.mark.parametrize(
        ("locale", "expected"),
        [
            ("hi", "hi-IN"),
            ("HI", "hi-IN"),
            ("ta", "ta-IN"),
            ("en", "en-IN"),
            # Sarvam spells Odia "od-IN", not the ISO-derived "or-IN"
            ("or", "od-IN"),
        ],
    )
    def test_maps_iso_locale_to_bcp47(self, locale: str, expected: str) -> None:
        assert resolve_tts_language_code(locale) == expected

    @pytest.mark.parametrize("locale", [None, ""])
    def test_missing_locale_defaults_to_en_in(self, locale: str | None) -> None:
        assert resolve_tts_language_code(locale) == "en-IN"

    def test_unmapped_locale_defaults_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert resolve_tts_language_code("xx") == "en-IN"

        assert "Unmapped locale" in caplog.text


class TestResolveDatasetTTSLanguageCode:
    def test_none_language_id_defaults_to_en_in_silently(
        self, db: Session, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert resolve_dataset_tts_language_code(db, None) == "en-IN"

        # An unset language is a normal dataset shape, not a misconfiguration.
        assert "Language not found" not in caplog.text

    def test_seeded_language_resolves(self, db: Session) -> None:
        odia = get_language_by_locale(session=db, locale="or")

        assert resolve_dataset_tts_language_code(db, odia.id) == "od-IN"

    def test_unknown_language_id_defaults_with_warning(
        self, db: Session, caplog: pytest.LogCaptureFixture
    ) -> None:
        missing_id = get_non_existent_id(db, Language)

        with caplog.at_level(logging.WARNING):
            assert resolve_dataset_tts_language_code(db, missing_id) == "en-IN"

        assert "Language not found" in caplog.text


class TestToElevenlabsLanguageCode:
    def test_known_bcp47_maps_to_iso(self) -> None:
        assert to_elevenlabs_language_code("hi-IN") == "hi"

    def test_odia_returns_none_for_auto_detect(self) -> None:
        assert to_elevenlabs_language_code("od-IN") is None
