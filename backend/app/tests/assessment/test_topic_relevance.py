"""Tests for the topic-relevance per-record request builder and result parser."""

import json
from unittest.mock import patch

from app.models.assessment import AssessmentAttachment
from app.services.assessment.prefilter import constants
from app.services.assessment.prefilter.topic_relevance import (
    build_topic_relevance_requests,
    parse_topic_relevance_results,
)


def _gemini():
    return patch.object(constants, "ASSESSMENT_PREFILTER_PROVIDER", "google")


class TestBuildRequests:
    def test_one_request_per_row_with_per_column_schema(self) -> None:
        rows = [(0, {"Problem": "p0"}), (1, {"Problem": "p1"})]
        with _gemini():
            lines = build_topic_relevance_requests(rows, ["Problem"], "rubric")
        assert [ln["key"] for ln in lines] == ["tr_0", "tr_1"]
        schema = lines[0]["request"]["generationConfig"]["responseSchema"]
        # per-column boolean + decision/reasoning
        assert schema["properties"]["Problem"]["type"] == "boolean"
        assert set(schema["required"]) == {"decision", "reasoning", "Problem"}
        assert "p0" in lines[0]["request"]["contents"][0]["parts"][0]["text"]

    def test_attachment_column_adds_part_and_schema_field(self) -> None:
        rows = [
            (0, {"Problem": "p0", "Docs": "https://drive.google.com/file/d/A/view"})
        ]
        atts = [AssessmentAttachment(column="Docs", type="image", format="url")]
        with _gemini():
            lines = build_topic_relevance_requests(rows, ["Problem"], "rubric", atts)
        schema = lines[0]["request"]["generationConfig"]["responseSchema"]
        assert "Docs" in schema["properties"]  # attachment column gets a verdict
        parts = lines[0]["request"]["contents"][0]["parts"]
        assert len(parts) >= 2  # text + at least one attachment part

    def test_empty_attachments_is_text_only(self) -> None:
        with _gemini():
            lines = build_topic_relevance_requests(
                [(0, {"Problem": "p"})], ["Problem"], "r"
            )
        assert len(lines[0]["request"]["contents"][0]["parts"]) == 1


class TestParseResults:
    def test_maps_decision_and_per_column_relevance(self) -> None:
        outputs = [
            {
                "row_id": "tr_0",
                "output": json.dumps(
                    {
                        "decision": "ACCEPT",
                        "reasoning": "ok",
                        "Problem": True,
                        "Docs": False,
                    }
                ),
                "error": None,
            },
            {
                "row_id": "tr_1",
                "output": json.dumps(
                    {"decision": "REJECT", "reasoning": "no", "Problem": False}
                ),
                "error": None,
            },
        ]
        parsed = parse_topic_relevance_results(outputs)
        assert parsed[0]["verdict"] is True
        assert parsed[0]["column_relevance"] == {"Problem": True, "Docs": False}
        assert parsed[1]["verdict"] is False
        assert parsed[1]["column_relevance"] == {"Problem": False}

    def test_unparseable_output_fails_open_accepted(self) -> None:
        # A gate response we cannot parse must NOT silently drop the submission:
        # it is accepted (verdict=True) so it still reaches L2 and is counted.
        parsed = parse_topic_relevance_results(
            [{"row_id": "tr_0", "output": "not json", "error": None}]
        )
        assert parsed[0]["verdict"] is True
        assert parsed[0]["decision"] == ""
        assert parsed[0]["reasoning"] == ""
        assert parsed[0]["column_relevance"] == {}

    def test_empty_output_fails_open_accepted(self) -> None:
        parsed = parse_topic_relevance_results(
            [{"row_id": "tr_0", "output": None, "error": "provider error"}]
        )
        assert parsed[0]["verdict"] is True
        assert parsed[0]["decision"] == ""
