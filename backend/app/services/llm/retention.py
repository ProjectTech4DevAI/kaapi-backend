import logging

from sqlmodel import Session

from app.core.config import settings
from app.core.util import now
from app.crud.llm import redact_llm_calls
from app.models.llm.response import LlmCallRedactionResult

logger = logging.getLogger(__name__)


def redact_aged_llm_calls(*, session: Session) -> LlmCallRedactionResult:
    cutoff = now() - settings.DELETE_ROLLING_WINDOW_TIMEDELTA

    logger.info(
        f"[redact_aged_llm_calls] Starting redaction | cutoff: {cutoff.isoformat()}"
    )

    rows_redacted = redact_llm_calls(session=session, cutoff=cutoff)

    logger.info(
        f"[redact_aged_llm_calls] Completed | rows_redacted: {rows_redacted} | "
        f"cutoff: {cutoff.isoformat()}"
    )

    return LlmCallRedactionResult(rows_redacted=rows_redacted, cutoff=cutoff)
