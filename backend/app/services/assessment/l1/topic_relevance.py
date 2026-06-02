"""Topic relevance filter for L1 pipeline.
"""

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from app.models.assessment import AssessmentAttachment
from app.services.assessment.utils.attachments import build_gemini_attachment_parts

logger = logging.getLogger(__name__)


def _build_output_schema(columns: list[str]) -> dict[str, Any]:
    """Build output schema: locked decision + per-column relevance booleans + reasoning."""
    props: dict[str, Any] = {
        "decision": {
            "type": "string",
            "enum": ["ACCEPT", "REJECT"],
            "description": "Final verdict. ACCEPT to proceed to full evaluation, REJECT to stop here.",
        },
    }
    required = ["decision"]

    for col in columns:
        props[col] = {
            "type": "boolean",
            "description": f"Whether the '{col}' column content is relevant to the topic.",
        }
        required.append(col)

    props["reasoning"] = {
        "type": "string",
        "description": "Explanation of the verdict and per-column relevance assessment.",
    }
    required.append("reasoning")

    return {"type": "object", "properties": props, "required": required}


def run_topic_relevance(
    row_idx: int,
    row: dict[str, str],
    columns: list[str],
    user_prompt: str,
    gemini_client: genai.Client,
    model: str,
    attachments: list[AssessmentAttachment] | None = None,
    type_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run topic relevance check on a single row.

    System instruction = user_prompt (the evaluation rubric/criteria).
    User content = the selected columns as JSON plus every mapped attachment
    (image/pdf) for the row, so relevance is judged on text and documents.
    Each attachment column also gets its own relevance boolean in the schema,
    so the export carries a ``topic_relevance_<doc_column>`` column.
    Output schema enforced: decision (ACCEPT/REJECT) + reasoning.
    On error defaults to verdict=True (fail-open).
    """
    # Document columns that actually have a value for this row.
    doc_columns: list[str] = []
    for att in attachments or []:
        if att.column not in doc_columns and (row.get(att.column) or "").strip():
            doc_columns.append(att.column)

    schema_columns = columns + doc_columns
    user_content = json.dumps({col: row.get(col, "") or "" for col in columns})
    output_schema = _build_output_schema(schema_columns)

    parts: list[dict[str, Any]] = [{"text": user_content}]
    for att in attachments or []:
        attachment_parts = build_gemini_attachment_parts(
            row.get(att.column, ""), att, type_cache
        )
        if attachment_parts:
            parts.append({"text": f"Attached document(s) for column '{att.column}':"})
            parts.extend(attachment_parts)

    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": parts}],
            config=types.GenerateContentConfig(
                system_instruction=user_prompt.strip(),
                response_mime_type="application/json",
                response_schema=output_schema,
                temperature=0.0,
            ),
        )
        raw = (response.text or "").strip()
        parsed = json.loads(raw)
        decision = str(parsed.get("decision", "ACCEPT")).upper()
        column_relevance = {col: bool(parsed.get(col, True)) for col in schema_columns}
        return {
            "row_id": f"row_{row_idx}",
            "verdict": decision == "ACCEPT",
            "decision": decision,
            "column_relevance": column_relevance,
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception as exc:
        logger.warning(
            "[run_topic_relevance] row_%s error — defaulting verdict=True | %s",
            row_idx,
            exc,
        )
        return {
            "row_id": f"row_{row_idx}",
            "verdict": True,
            "decision": "ACCEPT",
            "column_relevance": {col: True for col in schema_columns},
            "reasoning": f"(evaluation error — defaulting to pass) {exc}",
        }
