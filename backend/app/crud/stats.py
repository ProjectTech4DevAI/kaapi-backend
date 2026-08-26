import re

from sqlalchemy import text
from sqlmodel import Session

StatValue = str | int | float
StatRow = dict[str, StatValue]


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

_SIMPLE_COUNT = """
    SELECT
        o.name AS organization,
        p.name AS project,
        COUNT(*) FILTER (WHERE t.inserted_at >= now() - INTERVAL '24 hours')  AS count_24h,
        COUNT(*) AS count_7d
    FROM {table} t
    INNER JOIN organization o ON t.organization_id = o.id
    INNER JOIN project p ON t.project_id = p.id
    WHERE t.inserted_at >= now() - INTERVAL '168 hours'
    GROUP BY o.name, p.name
    ORDER BY count_7d DESC
"""

SIMPLE_COUNT_TABLES = {
    "Evaluation Runs": "evaluation_run",
    "STT Results": "stt_result",
    "TTS Results": "tts_result",
    "Assessments": "assessment",
}

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _simple_count_sql(table: str) -> str:
    if not _IDENTIFIER.match(table):
        raise ValueError(f"unsafe table identifier: {table!r}")
    return _SIMPLE_COUNT.format(table=table)


def _rows(session: Session, sql: str) -> list[StatRow]:
    result = session.connection().execute(text(sql))
    return [dict(row) for row in result.mappings().all()]


def get_daily_stats(*, session: Session) -> dict[str, list[StatRow]]:
    stats: dict[str, list[StatRow]] = {
        "LLM Calls": _rows(session, LLM_CALLS),
        "LLM Tokens": _rows(session, LLM_TOKENS),
        "LLM Modality": _rows(session, LLM_MODALITY),
        "Jobs by Type": _rows(session, JOBS),
    }
    for label, table in SIMPLE_COUNT_TABLES.items():
        stats[label] = _rows(session, _simple_count_sql(table))
    return stats
