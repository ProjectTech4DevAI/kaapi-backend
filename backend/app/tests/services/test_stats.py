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


def test_format_sections_renders_bold_title_and_aligned_table():
    sections = format_sections(_sample_stats())
    llm_section = next(s for s in sections if s.startswith("**LLM Calls**"))
    assert "organization  project  calls_24h  calls_7d" in llm_section
    assert "Acme          Alpha            3        15" in llm_section
    assert llm_section.count("```") == 2  # wrapped in one code block


def test_format_sections_marks_empty_sections():
    sections = format_sections(_sample_stats())
    stt_section = next(s for s in sections if s.startswith("**STT Results**"))
    assert stt_section == "**STT Results**\n_no data_"


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
    section = format_sections(stats)[0]
    assert "gemini-3.1-flash-…" in section  # clipped to 17 chars + ellipsis
    assert "gemini-3.1-flash-tts-preview" not in section
    assert "89,271" in section  # thousands separator applied


def test_post_to_discord_noop_when_webhook_unset():
    with patch.object(stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", None), patch(
        "app.services.stats.requests.post"
    ) as mock_post:
        post_to_discord(["anything"])
        mock_post.assert_not_called()


def test_post_to_discord_packs_sections_under_size_limit():
    posted: list[str] = []

    def fake_post(url, json, timeout):
        posted.append(json["content"])
        return MagicMock()  # provides raise_for_status()

    big_sections = ["x" * 1000 for _ in range(4)]
    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch("app.services.stats.requests.post", side_effect=fake_post):
        post_to_discord(big_sections)
    assert len(posted) >= 2  # split into multiple messages
    assert all(len(content) <= 2000 for content in posted)


def test_post_to_discord_swallows_request_exception():
    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch(
        "app.services.stats.requests.post",
        side_effect=requests.ConnectionError("boom"),
    ):
        post_to_discord(["hello"])  # must not raise


def test_post_to_discord_swallows_non_success_status():
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
    with patch.object(
        stats_mod.settings, "DISCORD_STATS_WEBHOOK_URL", "https://x/hook"
    ), patch("app.services.stats.requests.post", return_value=response):
        post_to_discord(["hello"])  # must not raise
    response.raise_for_status.assert_called_once()
