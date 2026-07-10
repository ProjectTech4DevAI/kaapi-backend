from unittest.mock import MagicMock, patch

import requests

from app.services import stats as stats_mod
from app.services.stats import (
    _chunk_message,
    collect_daily_stats,
    format_daily_stats_message,
    post_daily_stats_to_discord,
    section_counts,
)


def _sample_result() -> dict:
    return {
        "window": {
            "start_at": "2026-07-03T09:33:52.141126",
            "end_at": "2026-07-10T09:33:52.141126",
        },
        "stats": {
            "llm_call_counts": [
                {"organization_name": "Acme", "call_count": 15},
            ],
            "llm_call_token_summary": [
                {
                    "organization_name": "Acme",
                    "model": "gpt-4",
                    "sum_total_tokens": 3800,
                },
                {
                    "organization_name": "Beta",
                    "model": "claude-4",
                    "sum_total_tokens": 241,
                },
            ],
            "stt_result_counts": [],
            "tts_result_counts": [],
        },
    }


def test_section_counts_returns_row_len_per_list_section():
    counts = section_counts(_sample_result())
    assert counts == {
        "llm_call_counts": 1,
        "llm_call_token_summary": 2,
        "stt_result_counts": 0,
        "tts_result_counts": 0,
    }


def test_section_counts_skips_non_list_values():
    result = {"stats": {"a": [{"x": 1}], "b": "not-a-list", "c": None}}
    assert section_counts(result) == {"a": 1}


def test_format_message_header_shows_trimmed_window():
    msg = format_daily_stats_message(_sample_result())
    first_line = msg.splitlines()[0]
    assert first_line.startswith("Daily Stats")
    assert "2026-07-03 09:33:52" in first_line
    assert "2026-07-10 09:33:52" in first_line
    assert ".141126" not in first_line  # microseconds trimmed


def test_format_message_uses_friendly_section_titles_and_col_aliases():
    msg = format_daily_stats_message(_sample_result())
    assert "LLM Calls" in msg
    assert "LLM Tokens" in msg
    # column aliases applied
    assert "org" in msg
    assert "calls" in msg
    assert "tokens" in msg
    # raw column names should not appear
    assert "organization_name" not in msg
    assert "sum_total_tokens" not in msg


def test_format_message_collapses_empty_sections_into_one_line():
    msg = format_daily_stats_message(_sample_result())
    quiet_lines = [ln for ln in msg.splitlines() if ln.startswith("No data:")]
    assert len(quiet_lines) == 1
    assert "STT Results" in quiet_lines[0]
    assert "TTS Results" in quiet_lines[0]


def test_format_message_formats_numbers_with_commas():
    msg = format_daily_stats_message(_sample_result())
    assert "3,800" in msg
    assert "3800" not in msg


def test_format_message_truncates_long_sections():
    big = {
        "window": {"start_at": "2026-01-01T00:00:00", "end_at": "2026-01-08T00:00:00"},
        "stats": {
            "evaluation_run_counts": [
                {"organization_name": f"org-{i}", "row_count": i} for i in range(50)
            ]
        },
    }
    msg = format_daily_stats_message(big)
    assert "… +30 more" in msg  # 50 rows - 20 MAX = 30


def test_chunk_message_splits_on_section_boundaries_when_over_limit():
    # Each block ~40 chars — a few fit per 100-char chunk, forcing multiple chunks.
    blocks = [f"block-{i}-{'x' * 30}" for i in range(6)]
    message = "\n\n".join(blocks)
    chunks = _chunk_message(message, limit=100)
    assert len(chunks) > 1
    # Every original block appears somewhere in the chunks.
    joined = "\n\n".join(chunks)
    for b in blocks:
        assert b in joined


def test_chunk_message_single_chunk_when_under_limit():
    assert _chunk_message("short message", limit=1000) == ["short message"]


def test_post_to_discord_noop_when_webhook_unset():
    with patch.object(stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", None), patch(
        "app.services.stats.requests.post"
    ) as mock_post:
        post_daily_stats_to_discord("anything")
        mock_post.assert_not_called()


def test_post_to_discord_posts_each_chunk():
    posted: list[str] = []

    def fake_post(url, json, timeout):
        posted.append(json["content"])
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        return resp

    long_msg = "hdr\n\n" + "\n\n".join(f"block-{i}" * 100 for i in range(4))
    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch("app.services.stats.requests.post", side_effect=fake_post):
        post_daily_stats_to_discord(long_msg)
    assert len(posted) >= 2  # message chunked


def test_post_to_discord_swallows_request_exception():
    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch(
        "app.services.stats.requests.post", side_effect=requests.ConnectionError("boom")
    ):
        # must not raise
        post_daily_stats_to_discord("hello")


def test_post_to_discord_stops_after_first_failed_chunk():
    calls = {"n": 0}

    def flaky_post(url, json, timeout):
        calls["n"] += 1
        raise requests.ConnectionError("nope")

    long_msg = "\n\n".join(f"block-{i}" * 200 for i in range(4))
    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch("app.services.stats.requests.post", side_effect=flaky_post):
        post_daily_stats_to_discord(long_msg)
    assert calls["n"] == 1  # bailed after first failure, didn't retry rest


def test_collect_daily_stats_uses_default_window_when_hours_none():
    with patch(
        "app.services.stats.get_daily_stats", return_value={"x": []}
    ) as mock_get:
        result = collect_daily_stats(session=MagicMock())
    call_kwargs = mock_get.call_args.kwargs
    delta = call_kwargs["end_at"] - call_kwargs["start_at"]
    assert delta == stats_mod.DAILY_WINDOW
    assert result["stats"] == {"x": []}
    assert "start_at" in result["window"] and "end_at" in result["window"]


def test_collect_daily_stats_honours_window_hours_override():
    with patch("app.services.stats.get_daily_stats", return_value={}) as mock_get:
        collect_daily_stats(session=MagicMock(), window_hours=3)
    call_kwargs = mock_get.call_args.kwargs
    delta = call_kwargs["end_at"] - call_kwargs["start_at"]
    assert delta.total_seconds() == 3 * 3600
