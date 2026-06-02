"""Tests for prefilter topic relevance attachment handling."""

import json
from unittest.mock import MagicMock

from app.models.assessment import AssessmentAttachment
from app.services.assessment.prefilter.topic_relevance import run_topic_relevance


def _client_returning(decision: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.text = json.dumps(
        {"decision": decision, "Problem": True, "reasoning": "ok"}
    )
    client.models.generate_content.return_value = response
    return client


class TestTopicRelevanceAttachments:
    def test_attachments_added_to_contents(self) -> None:
        client = _client_returning("ACCEPT")
        att = AssessmentAttachment(column="Documents", type="image", format="url")
        row = {"Problem": "p", "Documents": "https://x.com/a/photo.jpg"}

        result = run_topic_relevance(
            row_idx=0,
            row=row,
            columns=["Problem"],
            user_prompt="rubric",
            gemini_client=client,
            model="gemini-2.5-flash",
            attachments=[att],
            type_cache={},
        )

        assert result["verdict"] is True
        contents = client.models.generate_content.call_args.kwargs["contents"]
        parts = contents[0]["parts"]
        # First part is the text JSON, then a label, then the attachment file part.
        assert parts[0]["text"]
        file_parts = [p for p in parts if "fileData" in p]
        assert len(file_parts) == 1
        assert file_parts[0]["fileData"]["fileUri"] == "https://x.com/a/photo.jpg"

    def test_document_relevance_in_schema_and_result(self) -> None:
        """Selected doc column gets its own relevance boolean in column_relevance."""
        client = MagicMock()
        response = MagicMock()
        response.text = json.dumps(
            {
                "decision": "ACCEPT",
                "Problem": True,
                "Documents": True,
                "reasoning": "ok",
            }
        )
        client.models.generate_content.return_value = response
        att = AssessmentAttachment(column="Documents", type="image", format="url")
        row = {"Problem": "p", "Documents": "https://x.com/a/photo.jpg"}

        result = run_topic_relevance(
            row_idx=3,
            row=row,
            columns=["Problem"],
            user_prompt="rubric",
            gemini_client=client,
            model="gemini-2.5-flash",
            attachments=[att],
            type_cache={},
        )

        # Document column carried into the per-column relevance map -> exports
        # as topic_relevance_Documents.
        assert "Documents" in result["column_relevance"]
        assert result["column_relevance"]["Documents"] is True
        schema = client.models.generate_content.call_args.kwargs[
            "config"
        ].response_schema
        assert "Documents" in schema["properties"]

    def test_no_attachments_text_only(self) -> None:
        client = _client_returning("REJECT")
        row = {"Problem": "p"}

        result = run_topic_relevance(
            row_idx=1,
            row=row,
            columns=["Problem"],
            user_prompt="rubric",
            gemini_client=client,
            model="gemini-2.5-flash",
        )

        assert result["verdict"] is False
        contents = client.models.generate_content.call_args.kwargs["contents"]
        parts = contents[0]["parts"]
        assert len(parts) == 1
        assert parts[0]["text"]

    def test_mixed_column_pdf_item_detected(self) -> None:
        client = _client_returning("ACCEPT")
        att = AssessmentAttachment(column="Documents", type="mixed", format="url")
        row = {"Problem": "p", "Documents": "https://x.com/a/report.pdf"}

        run_topic_relevance(
            row_idx=2,
            row=row,
            columns=["Problem"],
            user_prompt="rubric",
            gemini_client=client,
            model="gemini-2.5-flash",
            attachments=[att],
            type_cache={},
        )

        parts = client.models.generate_content.call_args.kwargs["contents"][0]["parts"]
        pdf_parts = [
            p
            for p in parts
            if p.get("fileData", {}).get("mimeType") == "application/pdf"
        ]
        assert len(pdf_parts) == 1
