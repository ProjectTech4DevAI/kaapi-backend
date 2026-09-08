import logging
from datetime import date
from typing import TypedDict

import requests

from app.core.config import settings
from app.crud.stats import StatRow, StatValue

logger = logging.getLogger(__name__)

DISCORD_EMBED_TOTAL_TEXT_LIMIT = (
    5900  # Discord caps all embed text in a message at 6000 chars.
)
DISCORD_EMBED_FIELD_VALUE_LIMIT = (
    1000  # Discord caps a single field value at 1024 chars.
)
DISCORD_EMBED_FIELD_COUNT_LIMIT = 25  # Discord embed field cap.
DISCORD_EMBED_BORDER_COLOR = 0x3B82F6  # Blue left-border accent on the Discord embed.
COLUMN_LABELS = {"24h": "Last 24hrs", "7d": "Last 7 days"}


class EmbedField(TypedDict):
    name: str
    value: str


class DiscordEmbed(TypedDict):
    title: str
    description: str
    color: int
    fields: list[EmbedField]


def _column_label(column: str) -> str:
    suffix = column.rsplit("_", 1)[-1]
    return COLUMN_LABELS.get(suffix, column)


def format_sections(stats: dict[str, list[StatRow]]) -> list[EmbedField]:
    """Build one Discord embed field per stat section (name = title, value = table)."""
    fields: list[EmbedField] = []
    inactive_titles: list[str] = []
    for title, rows in stats.items():
        if not rows:
            inactive_titles.append(title)
            continue

        columns = list(rows[0].keys())
        numeric_cols = [
            c for c in columns if all(isinstance(row[c], (int, float)) for row in rows)
        ]
        text_cols = [c for c in columns if c not in numeric_cols]

        # Repeating org/project on every row is what pushed rows past Discord's
        # embed width and wrapped the trailing columns. Instead, group rows by
        # every text column but the last one and print that group once as a
        # bold heading (bold only works outside the code block), leaving just
        # the varying column + metrics — short enough to fit — in the table.
        group_cols = text_cols[:-1]
        row_col = text_cols[-1] if text_cols else None
        table_cols = ([row_col] if row_col else []) + numeric_cols
        headers = {c: _column_label(c) for c in table_cols}

        def cell(column: str, row: StatRow) -> str:
            value = row[column]
            return f"{value:,}" if column in numeric_cols else str(value)

        groups: dict[tuple[StatValue, ...], list[StatRow]] = {}
        for row in rows:
            groups.setdefault(tuple(row[c] for c in group_cols), []).append(row)

        blocks = []
        for key, group_rows in groups.items():
            # Column width is driven by the widest value, uncapped, so long
            # names (e.g. model ids) are never truncated.
            widths = {
                c: max(len(headers[c]), max(len(cell(c, r)) for r in group_rows))
                for c in table_cols
            }

            def align(text: str, column: str) -> str:
                width = widths[column]
                return (
                    text.rjust(width) if column in numeric_cols else text.ljust(width)
                )

            lines = [
                "  ".join(align(headers[c], c) for c in table_cols).rstrip(),
            ]
            for row in group_rows:
                lines.append(
                    "  ".join(align(cell(c, row), c) for c in table_cols).rstrip()
                )

            table = "\n".join(lines)
            block = f"```\n{table}\n```"
            if group_cols:
                block = f"**{' / '.join(str(k) for k in key)}**\n{block}"
            blocks.append(block)

        value = "\n".join(blocks)
        if len(value) > DISCORD_EMBED_FIELD_VALUE_LIMIT:
            cutoff = DISCORD_EMBED_FIELD_VALUE_LIMIT - len("\n…\n```")
            truncated = value[:cutoff]
            value = (
                f"{truncated}\n…\n```"
                if truncated.count("```") % 2 == 1
                else f"{truncated}\n…"
            )
        fields.append({"name": title, "value": value})

    if inactive_titles:
        bullets = "\n".join(f"• {t}" for t in inactive_titles)
        fields.append({"name": "No activity this week", "value": bullets})
    return fields


def post_to_discord(fields: list[EmbedField], *, today: date | None = None) -> None:
    url = settings.DISCORD_STATS_WEBHOOK_URL
    if not url:
        return

    stat_date = today or date.today()
    title = f"Date: {stat_date.day}/{stat_date.month}/{stat_date.year}"
    description = "Daily platform feature stats"

    embed = _new_embed(title, description)
    for field in fields:
        field_cost = len(field["name"]) + len(field["value"])
        if embed["fields"] and (
            len(embed["fields"]) >= DISCORD_EMBED_FIELD_COUNT_LIMIT
            or _embed_size(embed) + field_cost > DISCORD_EMBED_TOTAL_TEXT_LIMIT
        ):
            _post(str(url), embed)
            embed = _new_embed(title, description)
        embed["fields"].append(field)
    if embed["fields"]:
        _post(str(url), embed)


def _new_embed(title: str, description: str) -> DiscordEmbed:
    return {
        "title": title,
        "description": description,
        "color": DISCORD_EMBED_BORDER_COLOR,
        "fields": [],
    }


def _embed_size(embed: DiscordEmbed) -> int:
    return (
        len(embed["title"])
        + len(embed["description"])
        + sum(len(f["name"]) + len(f["value"]) for f in embed["fields"])
    )


def _post(url: str, embed: DiscordEmbed) -> None:
    try:
        response = requests.post(url, json={"embeds": [embed]}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        # Log only the exception type — the message can contain the webhook URL.
        logger.warning(f"[_post] Webhook post failed: {type(e).__name__}")
