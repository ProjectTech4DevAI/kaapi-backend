import logging
from datetime import timedelta
from typing import Any

import requests
from sqlmodel import Session

from app.core.config import settings
from app.core.util import now
from app.crud.stats import get_daily_stats

logger = logging.getLogger(__name__)

DAILY_WINDOW = timedelta(hours=168)
_MAX_ROWS_PER_SECTION = 20
_DISCORD_CHUNK_LIMIT = 1900  # Discord content cap is 2000; leave headroom


def collect_daily_stats(
    *, session: Session, window_hours: int | None = None
) -> dict[str, Any]:
    end_at = now()
    start_at = end_at - (
        timedelta(hours=window_hours) if window_hours else DAILY_WINDOW
    )
    stats = get_daily_stats(session=session, start_at=start_at, end_at=end_at)
    return {
        "window": {
            "start_at": start_at.isoformat(),
            "end_at": end_at.isoformat(),
        },
        "stats": stats,
    }


def section_counts(result: dict[str, Any]) -> dict[str, int]:
    return {
        section: len(rows)
        for section, rows in result["stats"].items()
        if isinstance(rows, list)
    }


_COL_ALIASES = {
    "organization_name": "org",
    "sum_total_tokens": "tokens",
    "call_count": "calls",
    "job_count": "jobs",
    "row_count": "count",
}

_SECTION_TITLES = {
    "llm_call_counts": "LLM Calls",
    "llm_call_token_summary": "LLM Tokens",
    "llm_call_modality_counts": "LLM Modality",
    "job_type_counts": "Jobs by Type",
    "evaluation_run_counts": "Evaluation Runs",
    "stt_result_counts": "STT Results",
    "tts_result_counts": "TTS Results",
    "assessment_counts": "Assessments",
}


def _short_ts(iso: str) -> str:
    return iso.split(".", 1)[0].replace("T", " ")


def _fmt_num(v: Any) -> str:
    return f"{v:,}" if isinstance(v, int) else str(v)


def _render_section(section: str, rows: list[dict[str, Any]]) -> str | None:
    title = _SECTION_TITLES.get(section, section)
    if not rows:
        return None
    cols = list(rows[0].keys())
    sample = rows[:_MAX_ROWS_PER_SECTION]
    labels = {c: _COL_ALIASES.get(c, c) for c in cols}
    is_num = {c: all(isinstance(r.get(c), (int, float)) for r in sample) for c in cols}
    widths = {
        c: max(len(labels[c]), *(len(_fmt_num(r.get(c, ""))) for r in sample))
        for c in cols
    }

    def fmt(val: Any, c: str) -> str:
        s = _fmt_num(val)
        return f"{s:>{widths[c]}}" if is_num[c] else f"{s:<{widths[c]}}"

    header = "  ".join(fmt(labels[c], c) for c in cols)
    body = "\n".join("  ".join(fmt(r.get(c, ""), c) for c in cols) for r in sample)
    truncated = (
        f"\n… +{len(rows) - _MAX_ROWS_PER_SECTION} more"
        if len(rows) > _MAX_ROWS_PER_SECTION
        else ""
    )
    return f"\n{title}\n```\n{header}\n{body}{truncated}\n```"


def format_daily_stats_message(result: dict[str, Any]) -> str:
    window = result["window"]
    start = _short_ts(window["start_at"])
    end = _short_ts(window["end_at"])
    blocks = [f"Daily Stats  ·  {start} → {end} UTC"]

    quiet: list[str] = []
    for section, rows in result["stats"].items():
        if not isinstance(rows, list):
            continue
        rendered = _render_section(section, rows)
        if rendered is None:
            quiet.append(_SECTION_TITLES.get(section, section))
        else:
            blocks.append(rendered)

    if quiet:
        blocks.append("\nNo data: " + ", ".join(quiet))
    return "\n".join(blocks)


def _chunk_message(message: str, limit: int) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for block in message.split("\n\n"):
        if len(buf) + len(block) + 2 > limit and buf:
            chunks.append(buf)
            buf = block
        else:
            buf = f"{buf}\n\n{block}" if buf else block
    if buf:
        chunks.append(buf)
    return chunks


def post_daily_stats_to_discord(message: str) -> None:
    """Fire-and-forget Discord post. No-op if webhook not configured.
    Logs and stops on the first failed chunk — never raises."""
    url = settings.DISCORD_STATS_WEBHOOK_URL
    if not url:
        return
    for chunk in _chunk_message(message, _DISCORD_CHUNK_LIMIT):
        try:
            requests.post(
                str(url), json={"content": chunk}, timeout=5
            ).raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"[post_daily_stats_to_discord] Webhook post failed: {e}")
            return
