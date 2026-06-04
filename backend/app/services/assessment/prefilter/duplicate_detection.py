"""Duplicate detection stage: build per-record file_search batch requests, parse verdicts."""

import json
import logging
from typing import Any

from app.core.config import settings
from app.services.assessment.prefilter.request_builder import build_request_line

logger = logging.getLogger(__name__)

_DUP_SYS = """
You are a strict duplicate-detection judge for an innovation competition corpus.

If the submission is too vague for corpus matching (no problem/target/domain AND no
solution mechanism, or empty/gibberish), use verdict VAGUE.

Otherwise search the corpus and compare precisely. Focus on the MECHANISM of the
solution, not category or theme:
- DUPLICATE: problem AND solution mechanism substantially match a corpus entry.
- OVERLAP: problem OR solution mechanism matches; the other side clearly differs.
- PARTIAL_MATCH: thematic/conceptual similarity only — same domain, different mechanism.
- UNIQUE: neither problem nor solution substantially matches anything in the corpus.

Return JSON with keys: verdict, match_title, source_url, matching_sentence, reason.
For UNIQUE or VAGUE, set match_title, source_url and matching_sentence to "" and give a
short reason. Otherwise fill match_title, source_url (the SOURCE_URL verbatim from the
retrieved chunk), matching_sentence (the exact sentence) and a one-sentence reason.
Never invent or construct URLs or filenames.
"""

_DUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["DUPLICATE", "OVERLAP", "PARTIAL_MATCH", "UNIQUE", "VAGUE"],
        },
        "match_title": {"type": "string"},
        "source_url": {"type": "string"},
        "matching_sentence": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "verdict",
        "match_title",
        "source_url",
        "matching_sentence",
        "reason",
    ],
}


def _combined_text(row: dict[str, str], columns: list[str]) -> str:
    parts = [
        f"{col}:\n{row.get(col, '')}" for col in columns if row.get(col, "").strip()
    ]
    return "\n\n".join(parts) or "(empty submission)"


def build_duplicate_detection_requests(
    rows: list[tuple[int, dict[str, str]]],
    columns: list[str],
) -> list[dict[str, Any]]:
    """Build one batch JSONL line per record, grounded on the provider's corpus store."""
    store = settings.ASSESSMENT_PREFILTER_DUPLICATE_STORE or None
    return [
        build_request_line(
            key=f"dup_{idx}",
            system=_DUP_SYS,
            user_text=f"Submitted idea to check:\n\n{_combined_text(row, columns)}",
            response_schema=_DUP_SCHEMA,
            file_search_store=store,
        )
        for idx, row in rows
    ]


def parse_duplicate_detection_results(
    outputs: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Parse extracted batch outputs into {row_id: {verdict, match_title, ...}}."""
    parsed: dict[int, dict[str, Any]] = {}
    for out in outputs:
        key = str(out.get("row_id", ""))
        if not key.startswith("dup_"):
            continue
        try:
            idx = int(key.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if out.get("error") or not out.get("output"):
            parsed[idx] = _error_record(out.get("error") or "Empty response")
            continue
        try:
            data = json.loads(out["output"])
            parsed[idx] = {
                "verdict": str(data.get("verdict") or "UNKNOWN"),
                "match_title": data.get("match_title") or None,
                "source_url": data.get("source_url") or None,
                "matching_sentence": data.get("matching_sentence") or None,
                "reason": data.get("reason") or None,
            }
        except Exception as exc:
            logger.warning("[parse_duplicate_detection_results] %s — %s", key, exc)
            parsed[idx] = _error_record(str(exc)[:200])
    return parsed


def _error_record(reason: str) -> dict[str, Any]:
    return {
        "verdict": "ERROR",
        "match_title": None,
        "source_url": None,
        "matching_sentence": None,
        "reason": reason,
    }
