"""Tests for assessment/utils/export.py helper functions."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models.assessment import AssessmentExportRow
from app.services.assessment.utils import export as export_mod
from app.services.assessment.utils.export import (
    _build_export_row,
    _drop_empty_columns,
    _expand_input_columns,
    _expand_output_columns,
    _load_dataset_rows_for_run,
    _load_l2_results_for_run,
    _load_parsed_results_for_batch_job,
    _load_parsed_results_for_run,
    _load_prefilter_results,
    _safe_filename_part,
    _stage_batch_job,
    build_json_export_rows,
    load_export_rows_for_run,
    serialize_export_rows,
    sort_export_rows,
)
from app.models.assessment import Stage


def _run_ns(status: str = "processing") -> SimpleNamespace:
    return SimpleNamespace(
        id=5,
        assessment_id=9,
        status=status,
        config_id="00000000-0000-0000-0000-000000000001",
        config_version=1,
        updated_at=datetime(2026, 1, 1),
    )


def _assessment_ns() -> SimpleNamespace:
    return SimpleNamespace(experiment_name="exp", dataset_id=3)


class TestBuildExportRow:
    def test_prefilter_rejected_with_annotations(self) -> None:
        prefilter_item = {
            "prefilter_passed": False,
            "topic_relevance": {
                "decision": "REJECT",
                "reasoning": "off-topic",
                "column_relevance": {"Problem": False},
            },
            "duplicate_detection": {"row_id": "dup_0", "verdict": "UNIQUE"},
        }
        row = _build_export_row(
            run=_run_ns(),
            assessment=_assessment_ns(),
            dataset_name="ds",
            row_id="row_0",
            input_data={"Problem": "p"},
            prefilter_item=prefilter_item,
            l2_item=None,
            has_prefilter=True,
        )
        assert row.result_status == "prefilter_rejected"
        assert json.loads(row.topic_relevance)["decision"] == "REJECT"
        assert json.loads(row.duplicate_detection)["verdict"] == "UNIQUE"

    def test_passed_with_l2_output(self) -> None:
        row = _build_export_row(
            run=_run_ns(),
            assessment=_assessment_ns(),
            dataset_name=None,
            row_id="row_1",
            input_data=None,
            prefilter_item={"prefilter_passed": True},
            l2_item={"output": "{}", "error": None},
            has_prefilter=True,
        )
        assert row.result_status == "passed"

    def test_l2_error_is_failed_and_no_prefilter_cols(self) -> None:
        row = _build_export_row(
            run=_run_ns(),
            assessment=_assessment_ns(),
            dataset_name=None,
            row_id="row_2",
            input_data=None,
            prefilter_item=None,
            l2_item={"output": None, "error": "boom"},
            has_prefilter=False,
        )
        assert row.result_status == "failed"
        assert row.topic_relevance is None

    def test_no_l2_processing_vs_failed(self) -> None:
        processing = _build_export_row(
            run=_run_ns(status="processing"),
            assessment=_assessment_ns(),
            dataset_name=None,
            row_id="row_3",
            input_data=None,
            prefilter_item=None,
            l2_item=None,
            has_prefilter=False,
        )
        failed = _build_export_row(
            run=_run_ns(status="failed"),
            assessment=_assessment_ns(),
            dataset_name=None,
            row_id="row_4",
            input_data=None,
            prefilter_item=None,
            l2_item=None,
            has_prefilter=False,
        )
        assert processing.result_status == "processing"
        assert failed.result_status == "failed"


def _named_dataset() -> MagicMock:
    ds = MagicMock()
    ds.name = "ds"
    return ds


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

    def test_reserved_field_collision_namespaced(self) -> None:
        rows = [
            {
                "input_data": {"output": "expected answer", "question": "q1"},
                "output": "model answer",
            }
        ]
        expanded, keys = _expand_input_columns(rows)
        assert "input_output" in keys
        assert "question" in keys
        assert expanded[0]["input_output"] == "expected answer"
        assert expanded[0]["output"] == "model answer"


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
        expanded, fieldnames, *_ = _expand_output_columns(rows)
        assert "output" in fieldnames

    def test_json_dict_output_expanded(self) -> None:
        rows = [
            {"output": json.dumps({"score": 5, "reason": "good"}), "input_data": None}
        ]
        expanded, fieldnames, *_ = _expand_output_columns(rows)
        assert "score" in fieldnames
        assert "reason" in fieldnames
        assert expanded[0]["score"] == 5

    def test_mixed_parsed_and_unparsed_adds_output_raw(self) -> None:
        rows = [
            {"output": json.dumps({"score": 3}), "input_data": None},
            {"output": "not json", "input_data": None},
        ]
        expanded, fieldnames, *_ = _expand_output_columns(rows)
        assert "output_raw" in fieldnames
        # Second row that didn't parse should get output_raw
        assert expanded[1].get("output_raw") == "not json"

    def test_none_output_handled(self) -> None:
        rows = [{"output": None, "input_data": None}]
        expanded, fieldnames, *_ = _expand_output_columns(rows)
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
        lines = [line for line in payload.decode("utf-8").splitlines() if line.strip()]
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
    def test_sorts_by_config_version_then_numeric_row_index(self) -> None:
        rows = [
            _make_row(run_id=1, row_id="row_1", config_version=2),
            _make_row(run_id=2, row_id="row_0", config_version=1),
            _make_row(run_id=3, row_id="row_10", config_version=2),
            _make_row(run_id=4, row_id="row_2", config_version=2),
        ]
        sorted_rows = sort_export_rows(rows)
        assert sorted_rows[0].config_version == 1
        assert sorted_rows[1].config_version == 2
        assert [r.row_id for r in sorted_rows[1:]] == ["row_1", "row_2", "row_10"]

    def test_none_config_version_treated_as_zero(self) -> None:
        rows = [
            _make_row(run_id=1, row_id="row_0", config_version=1),
            _make_row(run_id=2, row_id="row_0", config_version=None),
        ]
        sorted_rows = sort_export_rows(rows)
        assert sorted_rows[0].config_version is None

    def test_invalid_row_id_suffix_falls_back_to_zero(self) -> None:
        rows = [
            _make_row(run_id=3, row_id="row_2", config_version=1),
            _make_row(run_id=2, row_id="row_xyz", config_version=1),
            _make_row(run_id=1, row_id="bad", config_version=1),
        ]
        sorted_rows = sort_export_rows(rows)
        assert [r.run_id for r in sorted_rows] == [1, 2, 3]

    def test_empty_list(self) -> None:
        assert sort_export_rows([]) == []


class TestExpandOutputColumnsDictOutput:
    def test_dict_output_expanded_directly(self) -> None:
        # raw output is already a dict (not a JSON string)
        rows = [{"output": {"score": 9, "label": "good"}, "input_data": None}]
        expanded, fieldnames, *_ = _expand_output_columns(rows)
        assert "score" in fieldnames
        assert expanded[0]["score"] == 9

    def test_non_dict_non_string_output_treated_as_unparsed(self) -> None:
        rows = [{"output": 42, "input_data": None}]
        expanded, fieldnames, *_ = _expand_output_columns(rows)
        # 42 is not a dict/string, treated as unparsed → output stays as-is
        assert "output" in fieldnames


class TestSerializeExportRowsXlsx:
    def test_xlsx_format_returns_xlsx_bytes(self) -> None:
        rows = [_make_row(output=json.dumps({"score": 3}))]
        payload, media_type = serialize_export_rows(rows, "xlsx")
        assert (
            media_type
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert len(payload) > 0

    def test_xlsx_no_excel_fields_falls_back_to_output(self) -> None:
        # Row with no output — excel_fields may be empty after filtering metadata
        rows = [_make_row(output=None)]
        _, media_type = serialize_export_rows(rows, "xlsx")
        assert (
            media_type
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


class TestBuildJsonExportRows:
    def test_returns_expanded_list(self) -> None:
        rows = [_make_row(output=json.dumps({"score": 7}))]
        result = build_json_export_rows(rows)
        assert isinstance(result, list)
        assert result[0]["score"] == 7

    def test_empty_input_returns_empty_list(self) -> None:
        assert build_json_export_rows([]) == []


class TestBuildExportResponse:
    def test_returns_streaming_response_with_disposition(self) -> None:
        from app.services.assessment.utils.export import build_export_response

        rows = [_make_row(output=json.dumps({"score": 3}))]
        with patch(
            "app.services.assessment.utils.export.generate_timestamped_filename",
            return_value="export_2024.csv",
        ):
            response = build_export_response(rows, "csv", "my experiment")

        assert response.media_type == "text/csv"
        assert "export_2024.csv" in response.headers["content-disposition"]

    def test_json_format_returns_json_response(self) -> None:
        from app.services.assessment.utils.export import build_export_response

        rows = [_make_row(output='{"score": 5}')]
        with patch(
            "app.services.assessment.utils.export.generate_timestamped_filename",
            return_value="export_2024.json",
        ):
            response = build_export_response(rows, "json", "exp")

        assert response.media_type == "application/json"


class TestLoadParsedResultsForRun:
    def _make_run(self, *, object_store_url: str | None = None) -> MagicMock:
        run = MagicMock()
        run.id = 1
        run.project_id = 1
        run.organization_id = 1
        run.object_store_url = object_store_url
        return run

    def _make_batch_job(
        self, *, provider: str = "openai", provider_output_file_id: str | None = None
    ) -> MagicMock:
        job = MagicMock()
        job.provider = provider
        job.provider_output_file_id = provider_output_file_id
        return job

    def test_no_url_no_file_id_returns_none(self) -> None:
        session = MagicMock()
        run = self._make_run()
        batch_job = self._make_batch_job()
        result = _load_parsed_results_for_run(
            session=session, run=run, batch_job=batch_job
        )
        assert result is None

    def test_s3_success_returns_parsed(self) -> None:
        session = MagicMock()
        run = self._make_run(object_store_url="s3://bucket/file.jsonl")
        batch_job = self._make_batch_job()

        raw_line = json.dumps(
            {
                "custom_id": "row_0",
                "response": {
                    "status_code": 200,
                    "body": {"output_text": "hello", "usage": {}},
                },
                "error": None,
            }
        )
        mock_body = MagicMock()
        mock_body.read.return_value = raw_line.encode()
        mock_storage = MagicMock()
        mock_storage.stream.return_value = mock_body

        with patch(
            "app.services.assessment.utils.export.get_cloud_storage",
            return_value=mock_storage,
        ):
            result = _load_parsed_results_for_run(
                session=session, run=run, batch_job=batch_job
            )

        assert result is not None
        assert result[0]["row_id"] == "row_0"

    def test_s3_failure_falls_back_to_none_when_no_file_id(self) -> None:
        session = MagicMock()
        run = self._make_run(object_store_url="s3://bucket/file.jsonl")
        batch_job = self._make_batch_job(provider_output_file_id=None)

        with patch(
            "app.services.assessment.utils.export.get_cloud_storage",
            side_effect=Exception("S3 down"),
        ):
            result = _load_parsed_results_for_run(
                session=session, run=run, batch_job=batch_job
            )

        assert result is None

    def test_s3_failure_falls_back_to_provider_download(self) -> None:
        session = MagicMock()
        run = self._make_run(object_store_url="s3://bucket/file.jsonl")
        batch_job = self._make_batch_job(
            provider="openai", provider_output_file_id="file_abc"
        )

        raw = [
            {
                "custom_id": "row_0",
                "response": {
                    "status_code": 200,
                    "body": {"output_text": "hi", "usage": {}},
                },
                "error": None,
            }
        ]
        with patch(
            "app.services.assessment.utils.export.get_cloud_storage",
            side_effect=Exception("S3 down"),
        ), patch(
            "app.services.assessment.utils.export._get_batch_provider",
            return_value=MagicMock(),
        ), patch(
            "app.services.assessment.utils.export.download_batch_results",
            return_value=raw,
        ):
            result = _load_parsed_results_for_run(
                session=session, run=run, batch_job=batch_job
            )

        assert result is not None
        assert result[0]["row_id"] == "row_0"

    def test_s3_empty_falls_back_logs_warning(self) -> None:
        session = MagicMock()
        run = self._make_run(object_store_url="s3://bucket/file.jsonl")
        batch_job = self._make_batch_job(provider_output_file_id=None)

        mock_body = MagicMock()
        mock_body.read.return_value = b""
        mock_storage = MagicMock()
        mock_storage.stream.return_value = mock_body

        with patch(
            "app.services.assessment.utils.export.get_cloud_storage",
            return_value=mock_storage,
        ):
            result = _load_parsed_results_for_run(
                session=session, run=run, batch_job=batch_job
            )

        assert result is None


class TestLoadDatasetRowsForRun:
    def _make_run(self) -> MagicMock:
        run = MagicMock()
        run.id = 1
        return run

    def _make_assessment(self, dataset_id: int = 1) -> MagicMock:
        assessment = MagicMock()
        assessment.id = 10
        assessment.dataset_id = dataset_id
        return assessment

    def test_dataset_not_found_returns_empty(self) -> None:
        session = MagicMock()
        session.get.return_value = None
        result = _load_dataset_rows_for_run(
            session=session, run=self._make_run(), assessment=self._make_assessment()
        )
        assert result == []

    def test_dataset_no_url_returns_empty(self) -> None:
        session = MagicMock()
        dataset = MagicMock()
        dataset.object_store_url = None
        session.get.return_value = dataset
        result = _load_dataset_rows_for_run(
            session=session, run=self._make_run(), assessment=self._make_assessment()
        )
        assert result == []

    def test_exception_returns_empty(self) -> None:
        session = MagicMock()
        session.get.side_effect = Exception("DB error")
        result = _load_dataset_rows_for_run(
            session=session, run=self._make_run(), assessment=self._make_assessment()
        )
        assert result == []

    def test_valid_dataset_returns_rows(self) -> None:
        session = MagicMock()
        dataset = MagicMock()
        dataset.object_store_url = "s3://bucket/ds.csv"
        session.get.return_value = dataset
        with patch(
            "app.services.assessment.utils.export._load_dataset_rows",
            return_value=[{"q": "hi"}],
        ):
            result = _load_dataset_rows_for_run(
                session=session,
                run=self._make_run(),
                assessment=self._make_assessment(),
            )
        assert result == [{"q": "hi"}]


class TestLoadExportRowsForRun:
    def _make_run(self) -> MagicMock:
        run = MagicMock()
        run.id = 1
        run.assessment_id = 10
        run.batch_job_id = 5
        run.status = "completed"
        run.config_id = None
        run.config_version = 1
        run.object_store_url = None
        run.updated_at = datetime(2024, 1, 1)
        return run

    def _make_assessment(self) -> MagicMock:
        assessment = MagicMock()
        assessment.id = 10
        assessment.experiment_name = "exp_v1"
        assessment.dataset_id = 2
        return assessment

    def _patches(self, *, l2, prefilter=None, dataset_rows=None):
        return [
            patch(
                "app.services.assessment.utils.export._load_l2_results_for_run",
                return_value=l2,
            ),
            patch(
                "app.services.assessment.utils.export._load_prefilter_results",
                return_value=prefilter or {},
            ),
            patch(
                "app.services.assessment.utils.export._load_dataset_rows_for_run",
                return_value=dataset_rows if dataset_rows is not None else [],
            ),
        ]

    def test_no_results_no_dataset_returns_empty(self) -> None:
        session = MagicMock()
        session.get.return_value = _named_dataset()
        run = self._make_run()
        p1, p2, p3 = self._patches(l2={})
        with p1, p2, p3:
            result = load_export_rows_for_run(
                session=session, run=run, assessment=self._make_assessment()
            )
        assert result == []

    def test_merged_results_build_export_rows(self) -> None:
        session = MagicMock()
        session.get.return_value = _named_dataset()
        run = self._make_run()
        l2 = {
            "row_0": {
                "row_id": "row_0",
                "output": '{"score": 5}',
                "error": None,
                "usage": None,
                "response_id": "r1",
            }
        }
        p1, p2, p3 = self._patches(l2=l2)
        with p1, p2, p3:
            result = load_export_rows_for_run(
                session=session, run=run, assessment=self._make_assessment()
            )
        assert len(result) == 1
        assert result[0].result_status == "passed"
        assert result[0].row_id == "row_0"

    def test_error_result_sets_failed_status(self) -> None:
        session = MagicMock()
        session.get.return_value = _named_dataset()
        run = self._make_run()
        l2 = {
            "row_0": {
                "row_id": "row_0",
                "output": None,
                "error": "timeout",
                "usage": None,
                "response_id": None,
            }
        }
        p1, p2, p3 = self._patches(l2=l2)
        with p1, p2, p3:
            result = load_export_rows_for_run(
                session=session, run=run, assessment=self._make_assessment()
            )
        assert result[0].result_status == "failed"

    def test_dataset_rows_include_pending_and_correlate_input(self) -> None:
        session = MagicMock()
        session.get.return_value = _named_dataset()
        run = self._make_run()
        run.status = "l2_processing"
        l2 = {
            "row_1": {
                "row_id": "row_1",
                "output": "x",
                "error": None,
                "usage": None,
                "response_id": None,
            }
        }
        dataset_rows = [{"q": "first"}, {"q": "second"}]
        p1, p2, p3 = self._patches(l2=l2, dataset_rows=dataset_rows)
        with p1, p2, p3:
            result = load_export_rows_for_run(
                session=session, run=run, assessment=self._make_assessment()
            )
        assert len(result) == 2
        assert result[0].result_status == "processing"  # row_0 not done yet
        assert result[1].input_data == {"q": "second"}
        assert result[1].result_status == "passed"


class TestStageBatchJob:
    def test_returns_job_for_stage(self) -> None:
        run = SimpleNamespace(stage_batches={Stage.L2_ASSESSMENT: 7})
        with patch.object(export_mod, "get_batch_job", return_value="JOB") as g:
            assert _stage_batch_job(MagicMock(), run, Stage.L2_ASSESSMENT) == "JOB"
        assert g.call_args.kwargs["batch_job_id"] == 7

    def test_none_when_no_batch(self) -> None:
        run = SimpleNamespace(stage_batches=None)
        assert _stage_batch_job(MagicMock(), run, Stage.L2_ASSESSMENT) is None


class TestLoadPrefilterResults:
    def test_merges_tr_and_dup_annotations(self) -> None:
        run = SimpleNamespace(id=5)
        assessment = SimpleNamespace(project_id=1)
        with patch.object(export_mod, "_stage_batch_job", return_value=SimpleNamespace(provider="openai")), patch.object(
            export_mod, "load_raw_batch_results", return_value=[]
        ), patch.object(export_mod, "parse_assessment_output", return_value=[]), patch.object(
            export_mod,
            "parse_topic_relevance_results",
            return_value={0: {"verdict": True, "decision": "ACCEPT", "reasoning": "ok", "column_relevance": {"a": True}}},
        ), patch.object(
            export_mod,
            "parse_duplicate_detection_results",
            return_value={0: {"verdict": "UNIQUE"}},
        ):
            out = _load_prefilter_results(MagicMock(), run, assessment)
        assert out["row_0"]["prefilter_passed"] is True
        assert out["row_0"]["topic_relevance"]["decision"] == "ACCEPT"
        assert out["row_0"]["duplicate_detection"]["verdict"] == "UNIQUE"

    def test_tr_load_failure_is_swallowed(self) -> None:
        run = SimpleNamespace(id=5)
        assessment = SimpleNamespace(project_id=1)
        with patch.object(export_mod, "_stage_batch_job", return_value=SimpleNamespace(provider="openai")), patch.object(
            export_mod, "load_raw_batch_results", side_effect=RuntimeError("s3 down")
        ):
            out = _load_prefilter_results(MagicMock(), run, assessment)
        assert out == {}


class TestLoadParsedResultsForBatchJob:
    def test_object_store_path(self) -> None:
        job = SimpleNamespace(id=1, provider="openai", raw_output_url="s3://x", provider_output_file_id=None)
        assessment = SimpleNamespace(project_id=1, organization_id=1)
        storage = MagicMock()
        storage.stream.return_value.read.return_value.decode.return_value = "raw"
        with patch.object(export_mod, "get_cloud_storage", return_value=storage), patch.object(
            export_mod, "parse_stored_results", return_value=[{"k": 1}]
        ), patch.object(export_mod, "parse_assessment_output", return_value=[{"row_id": "row_0"}]) as parse:
            result = _load_parsed_results_for_batch_job(MagicMock(), job, assessment)
        assert result == [{"row_id": "row_0"}]
        parse.assert_called_once()

    def test_provider_fallback_path(self) -> None:
        job = SimpleNamespace(id=1, provider="openai", raw_output_url=None, provider_output_file_id="f1", organization_id=1)
        assessment = SimpleNamespace(project_id=1, organization_id=1)
        with patch.object(export_mod, "_get_batch_provider", return_value=MagicMock()), patch.object(
            export_mod, "download_batch_results", return_value=[{"k": 1}]
        ), patch.object(export_mod, "parse_assessment_output", return_value=[{"row_id": "row_1"}]):
            result = _load_parsed_results_for_batch_job(MagicMock(), job, assessment)
        assert result == [{"row_id": "row_1"}]

    def test_returns_none_without_outputs(self) -> None:
        job = SimpleNamespace(id=1, provider="openai", raw_output_url=None, provider_output_file_id=None)
        assessment = SimpleNamespace(project_id=1, organization_id=1)
        assert _load_parsed_results_for_batch_job(MagicMock(), job, assessment) is None


class TestLoadL2ResultsForRun:
    def test_keys_by_row_id(self) -> None:
        run = SimpleNamespace()
        assessment = SimpleNamespace()
        with patch.object(export_mod, "_stage_batch_job", return_value=SimpleNamespace()), patch.object(
            export_mod,
            "_load_parsed_results_for_batch_job",
            return_value=[{"row_id": "row_0", "output": "x"}, {"no_row": 1}],
        ):
            merged = _load_l2_results_for_run(MagicMock(), run, assessment)
        assert set(merged) == {"row_0"}

    def test_empty_when_no_batch(self) -> None:
        with patch.object(export_mod, "_stage_batch_job", return_value=None):
            merged = _load_l2_results_for_run(MagicMock(), SimpleNamespace(), SimpleNamespace())
        assert merged == {}
