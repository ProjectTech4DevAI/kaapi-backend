from datetime import timedelta

from sqlmodel import Session

from app.core.config import settings
from app.core.util import now
from app.crud.llm import redact_llm_call_records
from app.models.llm.response import LlmCallRedactionResult


def redact_aged_llm_calls(*, session: Session) -> LlmCallRedactionResult:
    retention_cutoff = now() - timedelta(hours=settings.LLM_CALL_RETENTION_HOURS)
    rows_redacted = redact_llm_call_records(
        session=session, retention_cutoff=retention_cutoff
    )
    return LlmCallRedactionResult(rows_redacted=rows_redacted, cutoff=retention_cutoff)
