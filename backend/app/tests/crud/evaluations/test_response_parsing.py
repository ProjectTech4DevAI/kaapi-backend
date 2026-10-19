"""`response_parsing.py`: chunk shape is persisted to S3; bad input must not raise."""

from typing import Any

import pytest

from app.crud.evaluations.response_parsing import extract_file_search_chunks


def _file_search_call(results: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {"type": "file_search_call", "results": results}


class TestExtractFileSearchChunks:
    def test_flattens_hits_across_two_calls_in_payload_order(self) -> None:
        raw = {
            "output": [
                _file_search_call(
                    [
                        {"score": 0.91, "text": "chunk A", "filename": "a.pdf"},
                        {"score": 0.42, "text": "chunk B", "filename": "b.pdf"},
                    ]
                ),
                {"type": "message", "content": []},
                _file_search_call([{"score": 0.5, "text": "chunk C", "filename": "c"}]),
            ]
        }

        assert extract_file_search_chunks(raw) == [
            {"score": 0.91, "text": "chunk A", "filename": "a.pdf"},
            {"score": 0.42, "text": "chunk B", "filename": "b.pdf"},
            {"score": 0.5, "text": "chunk C", "filename": "c"},
        ]

    def test_hit_without_filename_yields_none(self) -> None:
        raw = {"output": [_file_search_call([{"score": 0.7, "text": "chunk"}])]}

        assert extract_file_search_chunks(raw) == [
            {"score": 0.7, "text": "chunk", "filename": None}
        ]

    def test_non_file_search_items_are_skipped(self) -> None:
        raw = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
                {"type": "reasoning"},
            ]
        }

        assert extract_file_search_chunks(raw) == []

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            {},
            {"output": None},
            {"output": []},
            {"output": [{"type": "file_search_call", "results": None}]},
            {"output": [{"type": "file_search_call"}]},
        ],
    )
    def test_malformed_or_empty_payloads_return_no_chunks(
        self, raw: dict[str, Any] | None
    ) -> None:
        assert extract_file_search_chunks(raw) == []
