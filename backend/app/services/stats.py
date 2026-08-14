import logging
from datetime import date
from typing import Any

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

EMBED_TOTAL_LIMIT = 5900  # Discord caps all embed text in a message at 6000 chars.
EMBED_FIELD_LIMIT = 1000  # Discord caps a single field value at 1024 chars.
MAX_FIELDS_PER_EMBED = 25  # Discord embed field cap.
MAX_COL_WIDTH = 18  # Cap each column so wide tables don't wrap in Discord.
BORDER_COLOR = 0x3B82F6  # Blue left-border accent on the Discord embed.


def _clip(text: str) -> str:
    if len(text) <= MAX_COL_WIDTH:
        return text
    return text[: MAX_COL_WIDTH - 1] + "…"


def format_sections(stats: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    """Build one Discord embed field per stat section (name = title, value = table)."""
    fields: list[dict[str, str]] = []
    for title, rows in stats.items():
        if not rows:
            fields.append({"name": title, "value": "_no data_"})
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
        value = f"```\n{table}\n```"
        if len(value) > EMBED_FIELD_LIMIT:
            cutoff = EMBED_FIELD_LIMIT - len("\n…\n```")
            value = f"{value[:cutoff]}\n…\n```"
        fields.append({"name": title, "value": value})
    return fields


def post_to_discord(fields: list[dict[str, str]], *, today: date | None = None) -> None:
    url = settings.DISCORD_STATS_WEBHOOK_URL
    if not url:
        return

    stat_date = today or date.today()
    title = f"Date: {stat_date.month}/{stat_date.day}/{stat_date.year}"
    description = "Daily platform feature stats"

    embed = _new_embed(title, description)
    for field in fields:
        field_cost = len(field["name"]) + len(field["value"])
        if embed["fields"] and (
            len(embed["fields"]) >= MAX_FIELDS_PER_EMBED
            or _embed_size(embed) + field_cost > EMBED_TOTAL_LIMIT
        ):
            _post(str(url), embed)
            embed = _new_embed(title, description)
        embed["fields"].append(field)
    if embed["fields"]:
        _post(str(url), embed)


def _new_embed(title: str, description: str) -> dict[str, Any]:
    return {
        "title": title,
        "description": description,
        "color": BORDER_COLOR,
        "fields": [],
    }


def _embed_size(embed: dict[str, Any]) -> int:
    return (
        len(embed["title"])
        + len(embed["description"])
        + sum(len(f["name"]) + len(f["value"]) for f in embed["fields"])
    )


def _post(url: str, embed: dict[str, Any]) -> None:
    try:
        response = requests.post(url, json={"embeds": [embed]}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        # Log only the exception type — the message can contain the webhook URL.
        logger.warning(f"[_post] Webhook post failed: {type(e).__name__}")
