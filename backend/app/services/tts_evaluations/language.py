"""Language-code resolution for TTS evaluation providers.

Datasets store an ISO 639-1 locale (`Language.locale`, e.g. "hi"); the canonical
code carried through the TTS pipeline is a BCP-47 tag (e.g. "hi-IN"), which is
what Sarvam takes directly and what ElevenLabs' ISO code is derived from.
"""

import logging

from sqlmodel import Session

from app.crud.language import get_language_locale_by_id
from app.models.llm.constants import BCP47_TO_ELEVENLABS_LANG
from app.services.tts_evaluations.constants import DEFAULT_TTS_LANGUAGE_CODE

logger = logging.getLogger(__name__)

# Sarvam spells Odia "od-IN" while ISO 639-1 is "or"; it is absent from the
# ElevenLabs map, so it can't be derived from the reverse lookup.
_ODIA_ISO_CODE = "or"
_ODIA_BCP47_CODE = "od-IN"

ISO_TO_BCP47_LANG: dict[str, str] = {
    iso_code: bcp47_code for bcp47_code, iso_code in BCP47_TO_ELEVENLABS_LANG.items()
}
ISO_TO_BCP47_LANG[_ODIA_ISO_CODE] = _ODIA_BCP47_CODE


def resolve_tts_language_code(locale: str | None) -> str:
    """Map an ISO 639-1 locale to the pipeline's BCP-47 tag, defaulting to en-IN."""
    if not locale:
        return DEFAULT_TTS_LANGUAGE_CODE

    language_code = ISO_TO_BCP47_LANG.get(locale.lower())
    if language_code is None:
        logger.warning(
            f"[resolve_tts_language_code] Unmapped locale, using default | "
            f"locale: {locale}, default: {DEFAULT_TTS_LANGUAGE_CODE}"
        )
        return DEFAULT_TTS_LANGUAGE_CODE

    return language_code


def resolve_dataset_tts_language_code(session: Session, language_id: int | None) -> str:
    """Resolve a dataset's `language_id` to a BCP-47 tag (en-IN when unset/unknown)."""
    if language_id is None:
        return DEFAULT_TTS_LANGUAGE_CODE

    locale = get_language_locale_by_id(session=session, language_id=language_id)
    if locale is None:
        logger.warning(
            f"[resolve_dataset_tts_language_code] Language not found, using default | "
            f"language_id: {language_id}, default: {DEFAULT_TTS_LANGUAGE_CODE}"
        )
    return resolve_tts_language_code(locale)


def to_elevenlabs_language_code(language_code: str) -> str | None:
    """ElevenLabs ISO 639-1 code, or None to let it auto-detect (e.g. Odia)."""
    return BCP47_TO_ELEVENLABS_LANG.get(language_code)
