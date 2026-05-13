"""Parsing utilities for assessment batch results."""

import json
from typing import Any


def parse_stored_results(raw_content: str) -> list[dict[str, Any]]:
    """Parse stored batch results from JSONL or JSON array."""
    content = raw_content.strip()
    if not content:
        return []

    if content.startswith("["):
        parsed = json.loads(content)
        return parsed if isinstance(parsed, list) else []

    return [json.loads(line) for line in content.splitlines() if line.strip()]


def usage_totals(usage: Any) -> tuple[int | None, int | None, int | None]:
    """Extract common token totals from provider usage payloads."""
    if not isinstance(usage, dict):
        return None, None, None

    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return input_tokens, output_tokens, total_tokens
