"""Tests for assessment/utils/export.py helper functions."""

import json
from datetime import datetime
from uuid import UUID

import pytest

from app.assessment.models import AssessmentExportRow
from app.assessment.utils.export import (
    _drop_empty_columns,
    _expand_input_columns,
    _expand_output_columns,
    _safe_filename_part,
    serialize_export_rows,
    sort_export_rows,
)


def _make_row(
    *,
    run_id: int = 1,
    row_id: str = "row_0",
    output: str | None = None,
    input_data: dict | None = None,
    result_status: str = "passed",
    config_version: int | None = None,
) -> AssessmentExportRow:
    return AssessmentExportRow(
        assessment_id=1,
        experiment_name="exp",
        dataset_id=1,
        dataset_name="ds",
        run_id=run_id,
        run_name="run",
        run_status="completed",
        config_id=None,
        config_version=config_version,
        row_id=row_id,
        result_status=result_status,
        input_data=input_data,
        output=output,
        error=None,
        response_id=None,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        updated_at=datetime(2024, 1, 1),
    )


class TestSafeFilenamePart:
    def test_alphanumeric_unchanged(self) -> None:
        assert _safe_filename_part("my_export") == "my_export"

    def test_spaces_replaced(self) -> None:
        result = _safe_filename_part("my export file")
        assert " " not in result

    def test_special_chars_replaced(self) -> None:
        result = _safe_filename_part("hello/world:test")
        assert "/" not in result
        assert ":" not in result

    def test_empty_string_returns_default(self) -> None:
        assert _safe_filename_part("") == "assessment_results"

    def test_only_special_chars_returns_default(self) -> None:
        assert _safe_filename_part("!!!") == "assessment_results"

    def test_preserves_dots_and_hyphens(self) -> None:
        result = _safe_filename_part("my-file.v2")
        assert "." in result
        assert "-" in result


class TestExpandInputColumns:
    def test_no_input_data_removes_key(self) -> None:
        rows = [{"output": "x", "input_data": None}]
        expanded, keys = _expand_input_columns(rows)
        assert keys == []
        assert "input_data" not in expanded[0]

    def test_input_data_dict_expanded(self) -> None:
        rows = [{"input_data": {"question": "q1", "context": "c1"}, "output": "x"}]
        expanded, keys = _expand_input_columns(rows)
        assert "question" in keys
        assert "context" in keys
        assert expanded[0]["question"] == "q1"
        assert expanded[0]["context"] == "c1"
        assert "input_data" not in expanded[0]

    def test_multiple_rows_union_of_keys(self) -> None:
        rows = [
            {"input_data": {"a": "1"}, "output": "x"},
            {"input_data": {"b": "2"}, "output": "y"},
        ]
        _, keys = _expand_input_columns(rows)
        assert "a" in keys
        assert "b" in keys

    def test_missing_key_in_row_gets_none(self) -> None:
        rows = [
            {"input_data": {"a": "1", "b": "2"}, "output": "x"},
            {"input_data": {"a": "3"}, "output": "y"},
        ]
        expanded, _ = _expand_input_columns(rows)
        assert expanded[1].get("b") is None


class TestDropEmptyColumns:
    def test_keeps_non_empty_columns(self) -> None:
        rows = [{"a": "val", "b": None}, {"a": "val2", "b": ""}]
        result_rows, result_fields = _drop_empty_columns(rows, ["a", "b"])
        assert "a" in result_fields
        assert "b" not in result_fields

    def test_no_change_when_all_have_values(self) -> None:
        rows = [{"a": "1", "b": "2"}]
        result_rows, result_fields = _drop_empty_columns(rows, ["a", "b"])
        assert result_fields == ["a", "b"]

    def test_all_empty_drops_all(self) -> None:
        rows = [{"a": None, "b": None}]
        _, result_fields = _drop_empty_columns(rows, ["a", "b"])
        assert result_fields == []


class TestExpandOutputColumns:
    def test_plain_string_output_not_expanded(self) -> None:
        rows = [{"output": "plain text", "input_data": None}]
        expanded, fieldnames = _expand_output_columns(rows)
        assert "output" in fieldnames

    def test_json_dict_output_expanded(self) -> None:
        rows = [
            {"output": json.dumps({"score": 5, "reason": "good"}), "input_data": None}
        ]
        expanded, fieldnames = _expand_output_columns(rows)
        assert "score" in fieldnames
        assert "reason" in fieldnames
        assert expanded[0]["score"] == 5

    def test_mixed_parsed_and_unparsed_adds_output_raw(self) -> None:
        rows = [
            {"output": json.dumps({"score": 3}), "input_data": None},
            {"output": "not json", "input_data": None},
        ]
        expanded, fieldnames = _expand_output_columns(rows)
        assert "output_raw" in fieldnames
        # Second row that didn't parse should get output_raw
        assert expanded[1].get("output_raw") == "not json"

    def test_none_output_handled(self) -> None:
        rows = [{"output": None, "input_data": None}]
        expanded, fieldnames = _expand_output_columns(rows)
        assert expanded[0].get("output") is None


class TestSerializeExportRows:
    def _make_rows(self) -> list[AssessmentExportRow]:
        return [
            _make_row(row_id="row_0", output=json.dumps({"score": 4})),
            _make_row(row_id="row_1", output=json.dumps({"score": 2})),
        ]

    def test_json_format_returns_json_bytes(self) -> None:
        rows = self._make_rows()
        payload, media_type = serialize_export_rows(rows, "json")
        assert media_type == "application/json"
        parsed = json.loads(payload)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_csv_format_returns_csv_bytes(self) -> None:
        rows = self._make_rows()
        payload, media_type = serialize_export_rows(rows, "csv")
        assert media_type == "text/csv"
        content = payload.decode("utf-8")
        assert "score" in content

    def test_csv_contains_all_rows(self) -> None:
        rows = self._make_rows()
        payload, _ = serialize_export_rows(rows, "csv")
        lines = [l for l in payload.decode("utf-8").splitlines() if l.strip()]
        assert len(lines) == 3  # header + 2 data rows

    def test_json_with_no_output(self) -> None:
        rows = [_make_row(output=None)]
        payload, media_type = serialize_export_rows(rows, "json")
        assert media_type == "application/json"
        parsed = json.loads(payload)
        assert len(parsed) == 1

    def test_csv_with_input_data(self) -> None:
        rows = [_make_row(input_data={"question": "What?", "context": "Some context"})]
        payload, _ = serialize_export_rows(rows, "csv")
        content = payload.decode("utf-8")
        assert "question" in content
        assert "context" in content


class TestSortExportRows:
    def test_sorts_by_config_version_then_row_id(self) -> None:
        rows = [
            _make_row(run_id=1, row_id="row_1", config_version=2),
            _make_row(run_id=2, row_id="row_0", config_version=1),
            _make_row(run_id=3, row_id="row_0", config_version=2),
        ]
        sorted_rows = sort_export_rows(rows)
        assert sorted_rows[0].config_version == 1
        assert sorted_rows[1].config_version == 2
        assert sorted_rows[1].row_id == "row_0"

    def test_none_config_version_treated_as_zero(self) -> None:
        rows = [
            _make_row(run_id=1, row_id="row_0", config_version=1),
            _make_row(run_id=2, row_id="row_0", config_version=None),
        ]
        sorted_rows = sort_export_rows(rows)
        assert sorted_rows[0].config_version is None

    def test_empty_list(self) -> None:
        assert sort_export_rows([]) == []
