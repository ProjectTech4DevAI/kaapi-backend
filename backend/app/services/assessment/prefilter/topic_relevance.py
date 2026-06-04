"""Topic relevance go/no-go gate: one batch request per row (text + attachments).

Each request returns a per-column relevance boolean for every text and attachment
column plus a final ACCEPT/REJECT verdict.
"""

import json
import logging
from typing import Any

from app.core.config import settings
from app.models.assessment import AssessmentAttachment
from app.services.assessment.prefilter.request_builder import build_request_line
from app.services.assessment.utils.attachments import (
    attachment_type_for_row,
    build_gemini_attachment_parts,
    resolve_attachment_values,
)

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "\n\nJudge whether this submission is relevant to the topic. For EACH listed "
    "column (including any attached document/image columns) set its value to true if "
    "that column's content is relevant to the topic, else false. Then give a final "
    "decision: ACCEPT if relevant enough to proceed, otherwise REJECT."
)


def _build_schema(columns: list[str]) -> dict[str, Any]:
    """Output schema: decision + reasoning + a boolean per column."""
    props: dict[str, Any] = {
        "decision": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
        "reasoning": {"type": "string"},
    }
    for col in columns:
        props[col] = {"type": "boolean"}
    return {
        "type": "object",
        "properties": props,
        "required": ["decision", "reasoning", *columns],
    }


def _record_text(row: dict[str, str], columns: list[str]) -> str:
    return "\n\n".join(f"{col}:\n{row.get(col, '') or ''}" for col in columns)


def build_topic_relevance_requests(
    rows: list[tuple[int, dict[str, str]]],
    columns: list[str],
    user_prompt: str,
    attachments: list[AssessmentAttachment] | None = None,
) -> list[dict[str, Any]]:
    """Build one batch JSONL line per row, with text columns + attachment parts."""
    attachments = attachments or []
    is_openai = settings.ASSESSMENT_PREFILTER_PROVIDER == "openai"
    schema = _build_schema(columns + [a.column for a in attachments])
    system = user_prompt.strip() + _INSTRUCTIONS

    lines: list[dict[str, Any]] = []
    for idx, row in rows:
        attachment_parts: list[dict[str, Any]] = []
        for att in attachments:
            cell = row.get(att.column, "")
            if not cell.strip():
                continue
            override = attachment_type_for_row(att, row)
            attachment_parts.extend(
                resolve_attachment_values(cell, att, type_override=override)
                if is_openai
                else build_gemini_attachment_parts(cell, att, type_override=override)
            )
        lines.append(
            build_request_line(
                key=f"tr_{idx}",
                system=system,
                user_text=_record_text(row, columns),
                attachment_parts=attachment_parts or None,
                response_schema=schema,
            )
        )
    return lines


def parse_topic_relevance_results(
    outputs: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Parse outputs into {row_id: {verdict, decision, reasoning, column_relevance}}."""
    parsed: dict[int, dict[str, Any]] = {}
    for out in outputs:
        key = str(out.get("row_id", ""))
        if not key.startswith("tr_"):
            continue
        try:
            idx = int(key.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        try:
            data = json.loads(out.get("output") or "")
            decision = str(data.get("decision", "ACCEPT")).upper()
            column_relevance = {
                k: bool(v)
                for k, v in data.items()
                if k not in ("decision", "reasoning")
            }
            parsed[idx] = {
                "verdict": decision == "ACCEPT",
                "decision": decision,
                "reasoning": str(data.get("reasoning", "")),
                "column_relevance": column_relevance,
            }
        except Exception as exc:
            logger.warning("[parse_topic_relevance_results] %s — %s", key, exc)
    return parsed
