"""Prefilter config helpers shared by the batch pipeline stages."""

from typing import Any


def resolve_prefilter_settings(prefilter_config: dict[str, Any]) -> dict[str, Any]:
    """Flatten the prefilter config into the values the stage builders need."""
    tr_config = prefilter_config.get("topic_relevance") or {}
    dup_config = prefilter_config.get("duplicate_detection") or {}

    tr_columns = tr_config.get("columns") or []
    tr_attachment_columns = tr_config.get("attachment_columns")
    tr_prompt = tr_config.get("prompt") or ""
    dup_columns = dup_config.get("columns") or []

    return {
        "tr_columns": tr_columns,
        "tr_prompt": tr_prompt,
        "tr_attachment_columns": tr_attachment_columns,
        "dup_columns": dup_columns,
        "tr_enabled": bool((tr_columns or tr_attachment_columns) and tr_prompt),
        "dup_enabled": bool(dup_columns),
    }
