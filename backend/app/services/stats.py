import logging
from typing import Any

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

DISCORD_LIMIT = 1900  # Discord caps message content at 2000; leave headroom.
MAX_COL_WIDTH = 18  # Cap each column so wide tables don't wrap in Discord.


def _clip(text: str) -> str:
    if len(text) <= MAX_COL_WIDTH:
        return text
    return text[: MAX_COL_WIDTH - 1] + "…"


def format_sections(stats: dict[str, list[dict[str, Any]]]) -> list[str]:
    sections: list[str] = []
    for title, rows in stats.items():
        if not rows:
            sections.append(f"**{title}**\n_no data_")
            continue

        columns = list(rows[0].keys())

        # Numeric columns get thousands separators and are right-aligned so they
        # read as a clean column; text columns stay left-aligned.
        numeric = {
            c: all(isinstance(row[c], (int, float)) for row in rows) for c in columns
        }

        def cell(column: str, row: dict[str, Any]) -> str:
            value = row[column]
            text = f"{value:,}" if numeric[column] else str(value)
            return _clip(text)

        widths = {}
        for column in columns:
            cell_lengths = [len(cell(column, row)) for row in rows]
            widths[column] = max(len(_clip(column)), max(cell_lengths))

        def align(text: str, column: str) -> str:
            width = widths[column]
            return text.rjust(width) if numeric[column] else text.ljust(width)

        header = "  ".join(align(_clip(column), column) for column in columns)
        lines = [header.rstrip()]
        for row in rows:
            line = "  ".join(align(cell(column, row), column) for column in columns)
            lines.append(line.rstrip())

        table = "\n".join(lines)
        sections.append(f"**{title}**\n```\n{table}\n```")
    return sections


def post_to_discord(sections: list[str]) -> None:
    url = settings.DISCORD_STATS_WEBHOOK_URL
    if not url:
        return

    # Pack whole sections into messages under Discord's size cap so no code
    # block is split across two posts.
    chunk = "Daily Stats · last 24h and 7d (UTC)"
    for section in sections:
        if len(chunk) + len(section) + 2 > DISCORD_LIMIT:
            _post(str(url), chunk)
            chunk = ""
        chunk = f"{chunk}\n\n{section}" if chunk else section
    if chunk:
        _post(str(url), chunk)


def _post(url: str, content: str) -> None:
    try:
        response = requests.post(url, json={"content": content}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        # Log only the exception type — the message can contain the webhook URL.
        logger.warning(f"[_post] Webhook post failed: {type(e).__name__}")
