"""Average OpenAI call latency for fast evaluation runs.
"""

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from app.models import EvaluationRun

logger = logging.getLogger(__name__)

MS_PER_SECOND = 1000.0
LATENCY_MS_DECIMALS = 1

STAGE_RESPONSE = "response"
STAGE_EMBEDDING = "embedding"
STAGE_JUDGE = "judge"

AVG_LATENCY_MS_KEY = "avg_latency_ms"
CALL_COUNT_KEY = "call_count"
# Unrounded running total; internal to the rollup, never persisted.
_SUM_MS_KEY = "sum_ms"


def _stage_entry(rows: Iterable[Mapping[str, Any]]) -> dict[str, float] | None:
    """Unrounded `{sum_ms, call_count}` over successful timed rows; None if none."""
    sum_ms = 0.0
    count = 0
    for row in rows:
        if row.get("failed"):
            continue
        latency_ms = row.get("latency_ms")
        if latency_ms is None:
            continue
        sum_ms += float(latency_ms)
        count += 1

    if count == 0:
        return None
    return {_SUM_MS_KEY: sum_ms, CALL_COUNT_KEY: count}


def _emit_entry(sum_ms: float, count: int) -> dict[str, Any]:
    return {
        AVG_LATENCY_MS_KEY: round(sum_ms / count, LATENCY_MS_DECIMALS),
        CALL_COUNT_KEY: count,
    }


def _preserved_entry(entry: Mapping[str, Any] | None) -> dict[str, float] | None:
    """Carry a persisted stage entry forward, with an approximate sum for the overall."""
    if not entry:
        return None
    count = int(entry[CALL_COUNT_KEY])
    if count == 0:
        return None
    avg_latency_ms = float(entry[AVG_LATENCY_MS_KEY])
    # Keep the stored avg so re-emitting doesn't drift it through float re-division.
    return {
        _SUM_MS_KEY: avg_latency_ms * count,
        CALL_COUNT_KEY: count,
        AVG_LATENCY_MS_KEY: avg_latency_ms,
    }


def _build_latency_dict(
    response: dict[str, float] | None,
    embedding: dict[str, float] | None,
    judge: dict[str, float] | None,
) -> dict[str, Any] | None:
    """Combine present stages into the `eval_run.latency` payload; None if none."""
    stages: dict[str, dict[str, float] | None] = {
        STAGE_RESPONSE: response,
        STAGE_EMBEDDING: embedding,
        STAGE_JUDGE: judge,
    }

    latency: dict[str, Any] = {}
    total_ms = 0.0
    total_count = 0
    for stage, entry in stages.items():
        if entry is None:
            continue
        stage_sum = entry[_SUM_MS_KEY]
        stage_count = int(entry[CALL_COUNT_KEY])
        if AVG_LATENCY_MS_KEY in entry:
            latency[stage] = {
                AVG_LATENCY_MS_KEY: entry[AVG_LATENCY_MS_KEY],
                CALL_COUNT_KEY: stage_count,
            }
        else:
            latency[stage] = _emit_entry(stage_sum, stage_count)
        total_ms += stage_sum
        total_count += stage_count

    if total_count == 0:
        return None
    latency.update(_emit_entry(total_ms, total_count))
    return latency


def attach_latency(
    *,
    eval_run: EvaluationRun,
    log_prefix: str,
    response_results: list[dict[str, Any]] | None = None,
    embedding_results: list[dict[str, Any]] | None = None,
    judge_results: list[dict[str, Any]] | None = None,
) -> None:
    """Recompute latency for the passed stage(s) onto `eval_run.latency`, never raising."""
    try:
        existing = eval_run.latency or {}
        passed: dict[str, list[dict[str, Any]] | None] = {
            STAGE_RESPONSE: response_results,
            STAGE_EMBEDDING: embedding_results,
            STAGE_JUDGE: judge_results,
        }

        entries: dict[str, dict[str, float] | None] = {}
        for stage, rows in passed.items():
            if rows is None:
                entries[stage] = _preserved_entry(existing.get(stage))
                continue
            entry = _stage_entry(rows)
            entries[stage] = entry

        latency = _build_latency_dict(
            response=entries[STAGE_RESPONSE],
            embedding=entries[STAGE_EMBEDDING],
            judge=entries[STAGE_JUDGE],
        )
        if latency is None:
            return
        # Reassign wholesale: in-place mutation of a JSONB dict isn't flushed.
        eval_run.latency = latency
    except Exception as latency_err:
        logger.warning(
            f"[attach_latency] {log_prefix} Failed to compute latency | {latency_err}"
        )
