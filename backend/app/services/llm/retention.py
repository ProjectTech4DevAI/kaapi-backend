import logging

from sqlmodel import Session

from app.core.config import settings
from app.core.util import now
from app.crud.llm import redact_llm_call_batch

logger = logging.getLogger(__name__)

# Rows updated per statement; small enough to keep each row-lock window short.
LLM_CALL_REDACTION_BATCH_SIZE = 2000


def redact_aged_llm_calls(*, session: Session) -> dict[str, int | str]:
    cutoff = now() - settings.DELETE_ROLLING_WINDOW_TIMEDELTA

    logger.info(
        f"[redact_aged_llm_calls] Starting redaction | cutoff: {cutoff.isoformat()} | "
        f"batch_size: {LLM_CALL_REDACTION_BATCH_SIZE}"
    )

    total_redacted = 0
    batches_run = 0

    while True:
        redacted = redact_llm_call_batch(
            session=session,
            cutoff=cutoff,
            batch_size=LLM_CALL_REDACTION_BATCH_SIZE,
        )
        if redacted == 0:
            break

        total_redacted += redacted
        batches_run += 1
        logger.info(
            f"[redact_aged_llm_calls] Batch complete | batch: {batches_run} | "
            f"rows: {redacted} | total_rows: {total_redacted}"
        )

    logger.info(
        f"[redact_aged_llm_calls] Completed | rows_redacted: {total_redacted} | "
        f"batches_run: {batches_run} | cutoff: {cutoff.isoformat()}"
    )

    return {
        "rows_redacted": total_redacted,
        "batches_run": batches_run,
        "cutoff": cutoff.isoformat(),
    }
