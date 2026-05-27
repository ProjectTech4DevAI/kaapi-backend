"""Topic relevance filter for L1 pipeline.
"""

import json
import logging
from typing import Any

from google import genai
from google.genai import types

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
) -> dict[str, Any]:
    """Run topic relevance check on a single row.

    System instruction = user_prompt (the evaluation rubric/criteria).
    User content = dict of {column_name: value} for the selected columns.
    Output schema enforced: decision (ACCEPT/REJECT) + reasoning.
    On error defaults to verdict=True (fail-open).
    """
    user_content = json.dumps({col: row.get(col, "") or "" for col in columns})
    output_schema = _build_output_schema(columns)

    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=user_content,
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
        column_relevance = {col: bool(parsed.get(col, True)) for col in columns}
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
            "column_relevance": {col: True for col in columns},
            "reasoning": f"(evaluation error — defaulting to pass) {exc}",
        }
