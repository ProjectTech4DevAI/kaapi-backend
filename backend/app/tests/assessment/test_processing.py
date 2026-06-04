"""Tests for assessment/processing.py pure functions."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.crud.assessment.processing import (
    _get_batch_provider,
    _sanitize_json_output,
    parse_assessment_output,
    process_run_batches,
)
from app.models.assessment import Stage, StageStatus


class TestSanitizeJsonOutput:
    def test_valid_json_unchanged(self) -> None:
        raw = '{"key": "value"}'
        assert _sanitize_json_output(raw) == raw

    def test_bare_newline_inside_string_escaped(self) -> None:
        raw = '{"text": "line1\nline2"}'
        result = _sanitize_json_output(raw)
        parsed = json.loads(result)
        assert parsed["text"] == "line1\nline2"

    def test_bare_tab_inside_string_escaped(self) -> None:
        raw = '{"text": "col1\tcol2"}'
        result = _sanitize_json_output(raw)
        parsed = json.loads(result)
        assert parsed["text"] == "col1\tcol2"

    def test_bare_carriage_return_escaped(self) -> None:
        raw = '{"text": "a\rb"}'
        result = _sanitize_json_output(raw)
        parsed = json.loads(result)
        assert parsed["text"] == "a\rb"

    def test_escaped_chars_outside_string_not_changed(self) -> None:
        raw = '{"a": 1, "b": 2}'
        assert _sanitize_json_output(raw) == raw

    def test_already_escaped_newline_not_double_escaped(self) -> None:
        raw = '{"text": "line1\\nline2"}'
        result = _sanitize_json_output(raw)
        assert result == raw

    def test_empty_string(self) -> None:
        assert _sanitize_json_output("") == ""


class TestParseAssessmentOutputOpenAI:
    def _make_result(self, custom_id: str, output_text: str) -> dict:
        return {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "id": "resp_abc",
                    "output_text": output_text,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
            "error": None,
        }

    def test_successful_result_parsed(self) -> None:
        raw = [self._make_result("row_0", "some output")]
        results = parse_assessment_output(raw, "openai")
        assert len(results) == 1
        assert results[0]["row_id"] == "row_0"
        assert results[0]["output"] == "some output"
        assert results[0]["error"] is None

    def test_error_in_result(self) -> None:
        raw = [
            {
                "custom_id": "row_1",
                "response": {"status_code": 200, "body": {}},
                "error": {"message": "rate limit exceeded"},
            }
        ]
        results = parse_assessment_output(raw, "openai")
        assert results[0]["error"] == "rate limit exceeded"
        assert results[0]["output"] is None

    def test_4xx_status_code_is_error(self) -> None:
        raw = [
            {
                "custom_id": "row_2",
                "response": {
                    "status_code": 400,
                    "body": {"error": {"message": "invalid request"}},
                },
                "error": None,
            }
        ]
        results = parse_assessment_output(raw, "openai")
        assert results[0]["error"] == "invalid request"

    def test_json_output_text_re_serialized(self) -> None:
        payload = {"score": 4, "reason": "good"}
        raw = [self._make_result("row_0", json.dumps(payload))]
        results = parse_assessment_output(raw, "openai")
        re_parsed = json.loads(results[0]["output"])
        assert re_parsed["score"] == 4

    def test_output_text_from_output_list(self) -> None:
        raw = [
            {
                "custom_id": "row_0",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "resp_abc",
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": "hello world"}
                                ],
                            }
                        ],
                    },
                },
                "error": None,
            }
        ]
        results = parse_assessment_output(raw, "openai")
        assert results[0]["output"] == "hello world"

    def test_empty_output_text_sets_error(self) -> None:
        raw = [self._make_result("row_0", "")]
        results = parse_assessment_output(raw, "openai")
        assert results[0]["error"] == "Empty response output"

    def test_sanitize_fallback_on_bad_json(self) -> None:
        # JSON with literal newline inside a string value
        bad_json = '{"text": "line1\nline2"}'
        raw = [self._make_result("row_0", bad_json)]
        results = parse_assessment_output(raw, "openai")
        # Should not raise; output should be a valid JSON string
        assert results[0]["output"] is not None

    def test_multiple_results(self) -> None:
        raw = [
            self._make_result("row_0", "out0"),
            self._make_result("row_1", "out1"),
        ]
        results = parse_assessment_output(raw, "openai")
        assert len(results) == 2
        assert results[1]["row_id"] == "row_1"

    def test_openai_native_provider_accepted(self) -> None:
        raw = [self._make_result("row_0", "out")]
        results = parse_assessment_output(raw, "openai-native")
        assert results[0]["output"] == "out"


class TestParseAssessmentOutputGoogle:
    def test_successful_google_result(self) -> None:
        from unittest.mock import patch

        with patch(
            "app.crud.assessment.processing.extract_text_from_response_dict",
            return_value="gemini output",
        ):
            raw = [
                {"key": "row_0", "response": {"text": "gemini output"}, "error": None}
            ]
            results = parse_assessment_output(raw, "google")

        assert results[0]["row_id"] == "row_0"
        assert results[0]["output"] == "gemini output"
        assert results[0]["error"] is None

    def test_google_error_result(self) -> None:
        raw = [{"key": "row_0", "response": None, "error": "quota exceeded"}]
        results = parse_assessment_output(raw, "google")
        assert results[0]["error"] == "quota exceeded"
        assert results[0]["output"] is None

    def test_google_empty_response(self) -> None:
        raw = [{"key": "row_0", "response": None, "error": None}]
        results = parse_assessment_output(raw, "google")
        assert results[0]["error"] == "Empty response"

    def test_google_empty_text_from_response(self) -> None:
        from unittest.mock import patch

        with patch(
            "app.crud.assessment.processing.extract_text_from_response_dict",
            return_value="",
        ):
            raw = [{"key": "row_0", "response": {"candidates": []}, "error": None}]
            results = parse_assessment_output(raw, "google")
        assert results[0]["output"] is None
        assert results[0]["error"] == "Empty response output"

    def test_google_native_provider_accepted(self) -> None:
        from unittest.mock import patch

        with patch(
            "app.crud.assessment.processing.extract_text_from_response_dict",
            return_value="out",
        ):
            raw = [{"key": "row_0", "response": {"x": 1}, "error": None}]
            results = parse_assessment_output(raw, "google-native")
        assert results[0]["output"] == "out"


class TestGetBatchProvider:
    def test_unsupported_provider_raises(self) -> None:
        session = MagicMock()
        with pytest.raises(ValueError, match="Unsupported batch provider"):
            _get_batch_provider(
                session=session,
                provider_name="anthropic",
                organization_id=1,
                project_id=1,
            )

    def test_openai_provider_returned(self) -> None:
        session = MagicMock()
        mock_client = MagicMock()
        with patch(
            "app.services.assessment.stages.get_openai_client", return_value=mock_client
        ), patch("app.services.assessment.stages.OpenAIBatchProvider") as mock_cls:
            _get_batch_provider(
                session=session,
                provider_name="openai",
                organization_id=1,
                project_id=1,
            )
        mock_cls.assert_called_once_with(client=mock_client)

    def test_google_provider_returned(self) -> None:
        session = MagicMock()
        mock_gemini = MagicMock()
        with patch("app.services.assessment.stages.GeminiClient") as mock_cls, patch(
            "app.services.assessment.stages.GeminiBatchProvider"
        ) as mock_batch_cls:
            mock_cls.from_credentials.return_value = mock_gemini
            _get_batch_provider(
                session=session,
                provider_name="google",
                organization_id=1,
                project_id=1,
            )
        mock_batch_cls.assert_called_once_with(client=mock_gemini.client)


class TestProcessRunBatches:
    def _parent(self):
        return SimpleNamespace(organization_id=1, project_id=1, experiment_name="exp")

    def _run(self):
        return SimpleNamespace(
            id=1,
            assessment_id=10,
            status="processing",
            stage=Stage.L2_ASSESSMENT,
            stage_status=StageStatus.PROCESSING,
            stage_batches={Stage.L2_ASSESSMENT: 5},
        )

    @pytest.mark.asyncio
    async def test_completes_stage_and_finalizes(self) -> None:
        session = MagicMock()
        session.get.return_value = self._parent()
        run = self._run()

        with patch(
            "app.crud.assessment.processing.get_batch_job", return_value=MagicMock()
        ), patch(
            "app.crud.assessment.processing._get_batch_provider",
            return_value=MagicMock(),
        ), patch(
            "app.crud.assessment.processing._poll_stage_outcome",
            return_value="completed",
        ), patch(
            "app.crud.assessment.processing.advance_or_finalize", return_value=None
        ) as advance, patch(
            "app.crud.assessment.processing.recompute_assessment_status"
        ):
            result = await process_run_batches(run=run, session=session)

        advance.assert_called_once()
        assert result["action"] == "processed"
        assert run.stage_status == StageStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_advances_and_dispatches_next_stage(self) -> None:
        session = MagicMock()
        session.get.return_value = self._parent()
        run = self._run()
        run.stage = Stage.PRE_FILTER_TOPIC_RELEVANCE
        run.stage_batches = {Stage.PRE_FILTER_TOPIC_RELEVANCE: 5}

        with patch(
            "app.crud.assessment.processing.get_batch_job", return_value=MagicMock()
        ), patch(
            "app.crud.assessment.processing._get_batch_provider",
            return_value=MagicMock(),
        ), patch(
            "app.crud.assessment.processing._poll_stage_outcome",
            return_value="completed",
        ), patch(
            "app.crud.assessment.processing._record_gate_stats"
        ) as gate_stats, patch(
            "app.crud.assessment.processing.advance_or_finalize",
            return_value=Stage.L2_ASSESSMENT,
        ), patch(
            "app.crud.assessment.processing.recompute_assessment_status"
        ), patch(
            "app.crud.assessment.processing.run_assessment_pipeline"
        ) as dispatch:
            result = await process_run_batches(run=run, session=session)

        gate_stats.assert_called_once()  # TR is a gate stage
        dispatch.delay.assert_called_once()
        assert result["action"] == "processed"

    @pytest.mark.asyncio
    async def test_no_change_while_in_progress(self) -> None:
        session = MagicMock()
        session.get.return_value = self._parent()
        run = self._run()

        with patch(
            "app.crud.assessment.processing.get_batch_job", return_value=MagicMock()
        ), patch(
            "app.crud.assessment.processing._get_batch_provider",
            return_value=MagicMock(),
        ), patch(
            "app.crud.assessment.processing._poll_stage_outcome",
            return_value="no_change",
        ):
            result = await process_run_batches(run=run, session=session)

        assert result["action"] == "no_change"

    @pytest.mark.asyncio
    async def test_failed_stage_fails_run(self) -> None:
        session = MagicMock()
        session.get.return_value = self._parent()
        run = self._run()

        with patch(
            "app.crud.assessment.processing.get_batch_job",
            return_value=MagicMock(error_message="boom"),
        ), patch(
            "app.crud.assessment.processing._get_batch_provider",
            return_value=MagicMock(),
        ), patch(
            "app.crud.assessment.processing._poll_stage_outcome", return_value="failed"
        ), patch(
            "app.crud.assessment.processing.update_assessment_run_status"
        ), patch(
            "app.crud.assessment.processing.recompute_assessment_status"
        ):
            result = await process_run_batches(run=run, session=session)

        assert result["action"] == "failed"
        # Failed stage preserved (so a resume knows where to restart); only status flips.
        assert run.stage == Stage.L2_ASSESSMENT
        assert run.stage_status == StageStatus.FAILED


class TestPollStageOutcome:
    def _job(self, **kw):
        base = dict(provider_status="completed", provider_output_file_id=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_all_failed_no_output_is_failed(self) -> None:
        from app.crud.assessment.processing import _poll_stage_outcome

        with patch(
            "app.crud.assessment.processing.poll_batch_status",
            return_value={
                "request_counts": {"completed": 0, "failed": 3},
                "error_file_id": "err",
            },
        ):
            outcome = _poll_stage_outcome(MagicMock(), MagicMock(), self._job())
        assert outcome == "failed"

    def test_no_output_not_ready_is_no_change(self) -> None:
        from app.crud.assessment.processing import _poll_stage_outcome

        with patch(
            "app.crud.assessment.processing.poll_batch_status",
            return_value={"request_counts": {"completed": 0, "failed": 0}},
        ):
            outcome = _poll_stage_outcome(MagicMock(), MagicMock(), self._job())
        assert outcome == "no_change"

    def test_output_ready_is_completed(self) -> None:
        from app.crud.assessment.processing import _poll_stage_outcome

        with patch(
            "app.crud.assessment.processing.poll_batch_status", return_value={}
        ), patch("app.crud.assessment.processing.process_completed_batch"):
            outcome = _poll_stage_outcome(
                MagicMock(), MagicMock(), self._job(provider_output_file_id="file_1")
            )
        assert outcome == "completed"
