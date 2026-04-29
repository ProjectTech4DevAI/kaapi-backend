"""Tests for assessment/utils/parsing.py."""

import json

import pytest

from app.assessment.utils.parsing import parse_stored_results, usage_totals


class TestParseStoredResults:
    def test_empty_string_returns_empty_list(self) -> None:
        assert parse_stored_results("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert parse_stored_results("   \n  ") == []

    def test_json_array_format(self) -> None:
        data = [
            {"row_id": "row_0", "output": "hello"},
            {"row_id": "row_1", "output": "world"},
        ]
        result = parse_stored_results(json.dumps(data))
        assert result == data

    def test_jsonl_single_object_parsed_as_one_entry(self) -> None:
        # Does NOT start with '[', so treated as a single JSONL line
        result = parse_stored_results(json.dumps({"key": "value"}))
        assert result == [{"key": "value"}]

    def test_jsonl_format(self) -> None:
        lines = [{"row_id": "row_0", "output": "a"}, {"row_id": "row_1", "output": "b"}]
        raw = "\n".join(json.dumps(line) for line in lines)
        result = parse_stored_results(raw)
        assert result == lines

    def test_jsonl_skips_blank_lines(self) -> None:
        line = {"row_id": "row_0", "output": "x"}
        raw = f"\n{json.dumps(line)}\n\n"
        result = parse_stored_results(raw)
        assert result == [line]

    def test_jsonl_single_line(self) -> None:
        line = {"k": "v"}
        result = parse_stored_results(json.dumps(line))
        assert result == [line]


class TestUsageTotals:
    def test_non_dict_returns_nones(self) -> None:
        assert usage_totals(None) == (None, None, None)
        assert usage_totals("string") == (None, None, None)
        assert usage_totals(42) == (None, None, None)

    def test_openai_style_keys(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        assert usage_totals(usage) == (10, 20, 30)

    def test_anthropic_style_keys(self) -> None:
        usage = {"input_tokens": 5, "output_tokens": 15}
        assert usage_totals(usage) == (5, 15, 20)

    def test_total_tokens_computed_when_missing(self) -> None:
        usage = {"input_tokens": 3, "output_tokens": 7}
        inp, out, total = usage_totals(usage)
        assert total == 10

    def test_explicit_total_tokens_not_overridden(self) -> None:
        usage = {"input_tokens": 3, "output_tokens": 7, "total_tokens": 99}
        _, _, total = usage_totals(usage)
        assert total == 99

    def test_missing_tokens_return_none(self) -> None:
        assert usage_totals({}) == (None, None, None)

    def test_partial_tokens_no_total_computed(self) -> None:
        usage = {"input_tokens": 5}
        inp, out, total = usage_totals(usage)
        assert inp == 5
        assert out is None
        assert total is None
