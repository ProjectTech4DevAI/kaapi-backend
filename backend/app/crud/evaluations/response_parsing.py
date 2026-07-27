"""Shared parsers for OpenAI Responses payloads across fast + judge evals.

`field_value` reads a field from either an SDK object (`getattr`) or a plain dict
(tests pass dicts), so both the response and judge stages walk one Responses text
extractor instead of maintaining two that can silently drift.
"""

from typing import Any


def field_value(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an object or dict (SDK object vs test dict), with a default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_response_text(response: Any) -> str:
    """Extract generated text, preferring `output_text` then walking `output`."""
    output_text = field_value(response, "output_text")
    if output_text:
        return output_text

    output = field_value(response, "output")
    if not output:
        return ""

    for item in output:
        if field_value(item, "type") != "message":
            continue
        for content in field_value(item, "content") or []:
            if field_value(content, "type") == "output_text":
                text = field_value(content, "text")
                if text:
                    return text
    return ""
