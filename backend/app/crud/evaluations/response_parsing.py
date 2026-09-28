"""Shared parsers for OpenAI Responses payloads across fast + judge evals.

`field_value` reads a field from either an SDK object (`getattr`) or a plain dict
(tests pass dicts), so both the response and judge stages walk one Responses text
extractor instead of maintaining two that can silently drift.
"""

from typing import Any

FILE_SEARCH_CALL_TYPE = "file_search_call"


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


def extract_file_search_chunks(raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten a Responses payload's file_search hits into JSON-safe chunk dicts.

    Reads the plain `model_dump()` dict that `include_provider_raw_response` hands
    back, unlike `services/response/response.py::get_file_search_results`, which
    needs the live SDK object.
    """
    chunks: list[dict[str, Any]] = []
    for item in field_value(raw, "output") or []:
        if field_value(item, "type") != FILE_SEARCH_CALL_TYPE:
            continue
        for hit in field_value(item, "results") or []:
            chunks.append(
                {
                    "score": field_value(hit, "score"),
                    "text": field_value(hit, "text"),
                    "filename": field_value(hit, "filename"),
                }
            )
    return chunks
