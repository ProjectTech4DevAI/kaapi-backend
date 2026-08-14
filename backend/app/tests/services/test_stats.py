from datetime import date
from unittest.mock import MagicMock, patch

import requests

from app.services import stats as stats_mod
from app.services.stats import format_sections, post_to_discord


def _sample_stats() -> dict:
    return {
        "LLM Calls": [
            {
                "organization": "Acme",
                "project": "Alpha",
                "calls_24h": 3,
                "calls_7d": 15,
            },
        ],
        "STT Results": [],
    }


def test_format_sections_renders_field_per_section_with_aligned_table():
    fields = format_sections(_sample_stats())
    llm_field = next(f for f in fields if f["name"] == "LLM Calls")
    assert "organization  project  calls_24h  calls_7d" in llm_field["value"]
    assert "Acme          Alpha            3        15" in llm_field["value"]
    assert llm_field["value"].count("```") == 2  # wrapped in one code block


def test_format_sections_marks_empty_sections():
    fields = format_sections(_sample_stats())
    stt_field = next(f for f in fields if f["name"] == "STT Results")
    assert stt_field["value"] == "_no data_"


def test_format_sections_clips_long_values_and_formats_numbers():
    stats = {
        "LLM Tokens": [
            {
                "organization": "Org",
                "model": "gemini-3.1-flash-tts-preview",  # 28 chars, over the cap
                "tokens_7d": 89271,
            },
        ],
    }
    value = format_sections(stats)[0]["value"]
    assert "gemini-3.1-flash-…" in value  # clipped to 17 chars + ellipsis
    assert "gemini-3.1-flash-tts-preview" not in value
    assert "89,271" in value  # thousands separator applied


def test_post_to_discord_noop_when_webhook_unset():
    with patch.object(stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", None), patch(
        "app.services.stats.requests.post"
    ) as mock_post:
        post_to_discord([{"name": "X", "value": "y"}])
        mock_post.assert_not_called()


def test_post_to_discord_sets_title_description_and_border_color():
    posted: list[dict] = []

    def fake_post(url, json, timeout):
        posted.append(json["embeds"][0])
        return MagicMock()

    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch("app.services.stats.requests.post", side_effect=fake_post):
        post_to_discord(
            [{"name": "LLM Calls", "value": "```\nx\n```"}],
            today=date(2026, 8, 10),
        )

    assert len(posted) == 1
    embed = posted[0]
    assert embed["title"] == "Date: 8/10/2026"
    assert embed["description"] == "Daily platform feature stats"
    assert embed["color"] == stats_mod.BORDER_COLOR
    assert embed["fields"] == [{"name": "LLM Calls", "value": "```\nx\n```"}]


def test_post_to_discord_splits_fields_across_embeds_under_size_limit():
    posted: list[dict] = []

    def fake_post(url, json, timeout):
        posted.append(json["embeds"][0])
        return MagicMock()

    big_fields = [{"name": f"Section {i}", "value": "x" * 2000} for i in range(4)]
    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch("app.services.stats.requests.post", side_effect=fake_post):
        post_to_discord(big_fields)

    assert len(posted) >= 2  # split into multiple embeds/messages
    for embed in posted:
        size = (
            len(embed["title"])
            + len(embed["description"])
            + sum(len(f["name"]) + len(f["value"]) for f in embed["fields"])
        )
        assert size <= stats_mod.EMBED_TOTAL_LIMIT


def test_post_to_discord_swallows_request_exception():
    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch(
        "app.services.stats.requests.post",
        side_effect=requests.ConnectionError("boom"),
    ):
        post_to_discord([{"name": "X", "value": "y"}])  # must not raise


def test_post_to_discord_swallows_non_success_status():
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch("app.services.stats.requests.post", return_value=response):
        post_to_discord([{"name": "X", "value": "y"}])  # must not raise
    response.raise_for_status.assert_called_once()
