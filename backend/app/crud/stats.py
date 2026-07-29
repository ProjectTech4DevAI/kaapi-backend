import logging
from datetime import datetime
from typing import Any

from sqlalchemy import TextClause, text
from sqlmodel import Session

logger = logging.getLogger(__name__)


_LLM_TOKEN_SUMMARY_SQL = text(
    """
    SELECT
        o.name AS organization_name,
        l.model AS model,
        COALESCE(SUM((l.usage->>'total_tokens')::INTEGER), 0) AS sum_total_tokens
    FROM llm_call l
    INNER JOIN organization o ON l.organization_id = o.id
    WHERE l.inserted_at BETWEEN :start_at AND :end_at
      AND l.deleted_at IS NULL
    GROUP BY o.name, l.model
    ORDER BY sum_total_tokens DESC
    """
)

_LLM_CALL_COUNTS_SQL = text(
    """
    SELECT
        o.name AS organization_name,
        COUNT(*) AS call_count
    FROM llm_call l
    INNER JOIN organization o ON l.organization_id = o.id
    WHERE l.inserted_at BETWEEN :start_at AND :end_at
      AND l.deleted_at IS NULL
    GROUP BY o.name
    ORDER BY call_count DESC
    """
)

_LLM_MODALITY_COUNTS_SQL = text(
    """
    SELECT
        o.name AS organization_name,
        CASE
            WHEN l.input_type = 'text'  AND l.output_type = 'text'  THEN 'TEXT'
            WHEN l.input_type = 'audio' AND l.output_type = 'text'  THEN 'STT'
            WHEN l.input_type = 'text'  AND l.output_type = 'audio' THEN 'TTS'
            ELSE 'OTHER'
        END AS modality,
        COUNT(*) AS call_count
    FROM llm_call l
    INNER JOIN organization o ON l.organization_id = o.id
    WHERE l.inserted_at BETWEEN :start_at AND :end_at
      AND l.deleted_at IS NULL
    GROUP BY o.name, modality
    ORDER BY o.name, modality
    """
)

_JOB_COUNTS_SQL = text(
    """
    SELECT
        o.name AS organization_name,
        j.job_type AS job_type,
        COUNT(*) AS job_count
    FROM job j
    INNER JOIN project p ON j.project_id = p.id
    INNER JOIN organization o ON p.organization_id = o.id
    WHERE j.inserted_at BETWEEN :start_at AND :end_at
    GROUP BY o.name, j.job_type
    ORDER BY o.name, j.job_type
    """
)


def _org_count_sql(table: str) -> TextClause:
    return text(
        f"""
        SELECT
            o.name AS organization_name,
            COUNT(*) AS row_count
        FROM {table} t
        INNER JOIN organization o ON t.organization_id = o.id
        WHERE t.inserted_at BETWEEN :start_at AND :end_at
        GROUP BY o.name
        ORDER BY row_count DESC
        """
    )


_EVAL_RUN_COUNTS_SQL = _org_count_sql("evaluation_run")
_STT_RESULT_COUNTS_SQL = _org_count_sql("stt_result")
_TTS_RESULT_COUNTS_SQL = _org_count_sql("tts_result")
_ASSESSMENT_COUNTS_SQL = _org_count_sql("assessment")


def _rows(
    session: Session, stmt: TextClause, params: dict[str, Any]
) -> list[dict[str, Any]]:
    return [dict(row) for row in session.execute(stmt, params).mappings().all()]


def get_daily_stats(
    *, session: Session, start_at: datetime, end_at: datetime
) -> dict[str, list[dict[str, Any]]]:
    params = {"start_at": start_at, "end_at": end_at}
    logger.info(
        f"[get_daily_stats] Collecting stats | start_at: {start_at.isoformat()}, "
        f"end_at: {end_at.isoformat()}"
    )
    return {
        "llm_call_counts": _rows(session, _LLM_CALL_COUNTS_SQL, params),
        "llm_call_token_summary": _rows(session, _LLM_TOKEN_SUMMARY_SQL, params),
        "llm_call_modality_counts": _rows(session, _LLM_MODALITY_COUNTS_SQL, params),
        "job_type_counts": _rows(session, _JOB_COUNTS_SQL, params),
        "evaluation_run_counts": _rows(session, _EVAL_RUN_COUNTS_SQL, params),
        "stt_result_counts": _rows(session, _STT_RESULT_COUNTS_SQL, params),
        "tts_result_counts": _rows(session, _TTS_RESULT_COUNTS_SQL, params),
        "assessment_counts": _rows(session, _ASSESSMENT_COUNTS_SQL, params),
    }
