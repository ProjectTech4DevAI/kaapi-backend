from typing import Any

from sqlalchemy import text
from sqlmodel import Session

# Every query reports two rolling windows in one pass: the last 24 hours and the
# last 7 days (168 hours), broken down per organization and per project. The
# 24h FILTER count is a subset of the rows the 7d WHERE already selected.

LLM_CALLS = """
    SELECT
        o.name AS organization,
        p.name AS project,
        COUNT(*) FILTER (WHERE l.inserted_at >= now() - INTERVAL '24 hours')  AS calls_24h,
        COUNT(*) AS calls_7d
    FROM llm_call l
    INNER JOIN organization o ON l.organization_id = o.id
    INNER JOIN project p ON l.project_id = p.id
    WHERE l.inserted_at >= now() - INTERVAL '168 hours'
      AND l.deleted_at IS NULL
    GROUP BY o.name, p.name
    ORDER BY calls_7d DESC
"""

LLM_TOKENS = """
    SELECT
        o.name AS organization,
        p.name AS project,
        l.model AS model,
        COALESCE(SUM((l.usage->>'total_tokens')::INTEGER)
            FILTER (WHERE l.inserted_at >= now() - INTERVAL '24 hours'), 0) AS tokens_24h,
        COALESCE(SUM((l.usage->>'total_tokens')::INTEGER), 0) AS tokens_7d
    FROM llm_call l
    INNER JOIN organization o ON l.organization_id = o.id
    INNER JOIN project p ON l.project_id = p.id
    WHERE l.inserted_at >= now() - INTERVAL '168 hours'
      AND l.deleted_at IS NULL
    GROUP BY o.name, p.name, l.model
    ORDER BY tokens_7d DESC
"""

LLM_MODALITY = """
    SELECT
        o.name AS organization,
        p.name AS project,
        CASE
            WHEN l.input_type = 'text'  AND l.output_type = 'text'  THEN 'TEXT'
            WHEN l.input_type = 'audio' AND l.output_type = 'text'  THEN 'STT'
            WHEN l.input_type = 'text'  AND l.output_type = 'audio' THEN 'TTS'
            ELSE 'OTHER'
        END AS modality,
        COUNT(*) FILTER (WHERE l.inserted_at >= now() - INTERVAL '24 hours')  AS calls_24h,
        COUNT(*) AS calls_7d
    FROM llm_call l
    INNER JOIN organization o ON l.organization_id = o.id
    INNER JOIN project p ON l.project_id = p.id
    WHERE l.inserted_at >= now() - INTERVAL '168 hours'
      AND l.deleted_at IS NULL
    GROUP BY o.name, p.name, modality
    ORDER BY o.name, p.name, modality
"""

JOBS = """
    SELECT
        o.name AS organization,
        p.name AS project,
        j.job_type AS job_type,
        COUNT(*) FILTER (WHERE j.inserted_at >= now() - INTERVAL '24 hours')  AS jobs_24h,
        COUNT(*) AS jobs_7d
    FROM job j
    INNER JOIN project p ON j.project_id = p.id
    INNER JOIN organization o ON p.organization_id = o.id
    WHERE j.inserted_at >= now() - INTERVAL '168 hours'
    GROUP BY o.name, p.name, j.job_type
    ORDER BY o.name, p.name, j.job_type
"""

EVALUATION_RUNS = """
    SELECT
        o.name AS organization,
        p.name AS project,
        COUNT(*) FILTER (WHERE t.inserted_at >= now() - INTERVAL '24 hours')  AS count_24h,
        COUNT(*) AS count_7d
    FROM evaluation_run t
    INNER JOIN organization o ON t.organization_id = o.id
    INNER JOIN project p ON t.project_id = p.id
    WHERE t.inserted_at >= now() - INTERVAL '168 hours'
    GROUP BY o.name, p.name
    ORDER BY count_7d DESC
"""

STT_RESULTS = """
    SELECT
        o.name AS organization,
        p.name AS project,
        COUNT(*) FILTER (WHERE t.inserted_at >= now() - INTERVAL '24 hours')  AS count_24h,
        COUNT(*) AS count_7d
    FROM stt_result t
    INNER JOIN organization o ON t.organization_id = o.id
    INNER JOIN project p ON t.project_id = p.id
    WHERE t.inserted_at >= now() - INTERVAL '168 hours'
    GROUP BY o.name, p.name
    ORDER BY count_7d DESC
"""

TTS_RESULTS = """
    SELECT
        o.name AS organization,
        p.name AS project,
        COUNT(*) FILTER (WHERE t.inserted_at >= now() - INTERVAL '24 hours')  AS count_24h,
        COUNT(*) AS count_7d
    FROM tts_result t
    INNER JOIN organization o ON t.organization_id = o.id
    INNER JOIN project p ON t.project_id = p.id
    WHERE t.inserted_at >= now() - INTERVAL '168 hours'
    GROUP BY o.name, p.name
    ORDER BY count_7d DESC
"""

ASSESSMENTS = """
    SELECT
        o.name AS organization,
        p.name AS project,
        COUNT(*) FILTER (WHERE t.inserted_at >= now() - INTERVAL '24 hours')  AS count_24h,
        COUNT(*) AS count_7d
    FROM assessment t
    INNER JOIN organization o ON t.organization_id = o.id
    INNER JOIN project p ON t.project_id = p.id
    WHERE t.inserted_at >= now() - INTERVAL '168 hours'
    GROUP BY o.name, p.name
    ORDER BY count_7d DESC
"""


def _rows(session: Session, sql: str) -> list[dict[str, Any]]:
    result = session.connection().execute(text(sql))
    return [dict(row) for row in result.mappings().all()]


def get_daily_stats(*, session: Session) -> dict[str, list[dict[str, Any]]]:
    stats: dict[str, list[dict[str, Any]]] = {}
    stats["LLM Calls"] = _rows(session, LLM_CALLS)
    stats["LLM Tokens"] = _rows(session, LLM_TOKENS)
    stats["LLM Modality"] = _rows(session, LLM_MODALITY)
    stats["Jobs by Type"] = _rows(session, JOBS)
    stats["Evaluation Runs"] = _rows(session, EVALUATION_RUNS)
    stats["STT Results"] = _rows(session, STT_RESULTS)
    stats["TTS Results"] = _rows(session, TTS_RESULTS)
    stats["Assessments"] = _rows(session, ASSESSMENTS)
    return stats
