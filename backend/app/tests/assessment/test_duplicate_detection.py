"""Tests for L1 duplicate detection."""

import json
from unittest.mock import MagicMock

from app.services.assessment.l1.duplicate_detection import (
    _build_combined,
    _parse_verdict,
    run_duplicate_detection,
)


def _vague_client(vague: bool, reason: str = "r") -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.text = json.dumps({"vague": vague, "reason": reason})
    client.models.generate_content.return_value = resp
    return client


class TestBuildCombined:
    def test_joins_non_empty(self) -> None:
        out = _build_combined({"Problem": "p", "Solution": "s", "Empty": "  "})
        assert "Problem:\np" in out
        assert "Solution:\ns" in out
        assert "Empty" not in out


class TestParseVerdict:
    def test_full_fields(self) -> None:
        raw = (
            "Verdict: DUPLICATE\n"
            "Title: Some Idea\n"
            "Source: https://x.com/a\n"
            "URL: https://x.com/a\n"
            "Matching sentence: a beam alarm\n"
            "Reason: same mechanism"
        )
        out = _parse_verdict(raw)
        assert out["verdict"] == "DUPLICATE"
        assert out["match_title"] == "Some Idea"
        assert out["source_url"] == "https://x.com/a"
        assert out["matching_sentence"] == "a beam alarm"
        assert out["reason"] == "same mechanism"

    def test_unique_verdict_only(self) -> None:
        out = _parse_verdict("Verdict: UNIQUE\nReason: nothing matches")
        assert out["verdict"] == "UNIQUE"
        assert out["match_title"] is None

    def test_regex_fallback_when_key_missing(self) -> None:
        out = _parse_verdict("The result is clearly OVERLAP here.")
        assert out["verdict"] == "OVERLAP"

    def test_no_verdict_stays_empty(self) -> None:
        out = _parse_verdict("no decision present")
        assert out["verdict"] == ""


class TestRunDuplicateDetection:
    def test_vague_short_circuits(self) -> None:
        client = _vague_client(True, "too vague")
        result = run_duplicate_detection(
            row_idx=0,
            row={"Problem": "x"},
            columns=["Problem"],
            gemini_client=client,
            model="gemini-2.5-flash",
            store_name="store",
        )
        assert result["verdict"] == "VAGUE"
        assert result["reason"] == "too vague"
        # Only the vague check is called; no file-search second call.
        assert client.models.generate_content.call_count == 1

    def test_not_vague_runs_file_search(self) -> None:
        client = MagicMock()
        vague_resp = MagicMock()
        vague_resp.text = json.dumps({"vague": False, "reason": ""})
        search_resp = MagicMock()
        search_resp.text = "Verdict: UNIQUE\nReason: novel"
        client.models.generate_content.side_effect = [vague_resp, search_resp]

        result = run_duplicate_detection(
            row_idx=1,
            row={"Problem": "p", "Solution": "s"},
            columns=["Problem", "Solution"],
            gemini_client=client,
            model="gemini-2.5-flash",
            store_name="store",
        )
        assert result["verdict"] == "UNIQUE"
        assert result["reason"] == "novel"
        assert result["row_id"] == "row_1"

    def test_file_search_error_returns_error_verdict(self) -> None:
        client = MagicMock()
        vague_resp = MagicMock()
        vague_resp.text = json.dumps({"vague": False, "reason": ""})
        client.models.generate_content.side_effect = [
            vague_resp,
            RuntimeError("search boom"),
        ]

        result = run_duplicate_detection(
            row_idx=2,
            row={"Problem": "p"},
            columns=["Problem"],
            gemini_client=client,
            model="gemini-2.5-flash",
            store_name="store",
        )
        assert result["verdict"] == "ERROR"
        assert "search boom" in result["reason"]

    def test_vague_check_parse_error_defaults_not_vague(self) -> None:
        client = MagicMock()
        bad_vague = MagicMock()
        bad_vague.text = "not json"
        search_resp = MagicMock()
        search_resp.text = "Verdict: PARTIAL_MATCH\nTitle: T\nReason: theme"
        client.models.generate_content.side_effect = [bad_vague, search_resp]

        result = run_duplicate_detection(
            row_idx=3,
            row={"Problem": "p"},
            columns=["Problem"],
            gemini_client=client,
            model="gemini-2.5-flash",
            store_name="store",
        )
        assert result["verdict"] == "PARTIAL_MATCH"
