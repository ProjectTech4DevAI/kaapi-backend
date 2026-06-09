"""Tests for the duplicate-detection batch request builder and result parser."""

from unittest.mock import patch

from app.services.assessment.prefilter import constants
from app.services.assessment.prefilter.duplicate_detection import (
    build_duplicate_detection_requests,
    parse_duplicate_detection_results,
)


class TestBuildRequests:
    def test_one_request_per_record(self) -> None:
        rows = [(0, {"Problem": "p0", "Solution": "s0"}), (1, {"Problem": "p1"})]
        lines = build_duplicate_detection_requests(rows, ["Problem", "Solution"])
        # key (gemini) or custom_id (openai) depending on configured provider.
        keys = [ln.get("key") or ln.get("custom_id") for ln in lines]
        assert keys == ["dup_0", "dup_1"]

    def test_openai_request_grounds_on_file_search_store(self) -> None:
        with patch.object(
            constants, "ASSESSMENT_PREFILTER_PROVIDER", "openai"
        ), patch.object(constants, "ASSESSMENT_PREFILTER_DUPLICATE_STORE", "vs_corpus"):
            lines = build_duplicate_detection_requests(
                [(0, {"Problem": "p"})], ["Problem"]
            )
        tool = lines[0]["body"]["tools"][0]
        assert tool["type"] == "file_search"
        assert tool["vector_store_ids"] == ["vs_corpus"]


class TestParseResults:
    def test_parses_structured_verdict_per_row(self) -> None:
        import json

        outputs = [
            {
                "row_id": "dup_0",
                "output": json.dumps(
                    {
                        "verdict": "UNIQUE",
                        "match_title": "",
                        "source_url": "",
                        "matching_sentence": "",
                        "reason": "novel",
                    }
                ),
                "error": None,
            },
            {
                "row_id": "dup_1",
                "output": json.dumps(
                    {
                        "verdict": "DUPLICATE",
                        "match_title": "T",
                        "source_url": "http://x",
                        "matching_sentence": "s",
                        "reason": "same mechanism",
                    }
                ),
                "error": None,
            },
        ]
        parsed = parse_duplicate_detection_results(outputs)
        assert parsed[0]["verdict"] == "UNIQUE"
        assert parsed[0]["source_url"] is None  # "" -> None
        assert parsed[1]["verdict"] == "DUPLICATE"
        assert parsed[1]["source_url"] == "http://x"

    def test_empty_response_records_error(self) -> None:
        parsed = parse_duplicate_detection_results(
            [{"row_id": "dup_3", "output": None, "error": None}]
        )
        assert parsed[3]["verdict"] == "ERROR"

    def test_bad_json_records_error_and_foreign_keys_skipped(self) -> None:
        parsed = parse_duplicate_detection_results(
            [
                {"row_id": "tr_0", "output": "{}", "error": None},  # not a dup key
                {"row_id": "dup_x", "output": "{}", "error": None},  # bad index
                {"row_id": "dup_4", "output": "{not json", "error": None},
            ]
        )
        assert set(parsed) == {4}
        assert parsed[4]["verdict"] == "ERROR"
