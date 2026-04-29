"""Tests for assessment/processing.py pure functions."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.assessment.processing import (
    _get_batch_provider,
    _sanitize_json_output,
    parse_assessment_output,
    poll_all_pending_assessments,
)


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
            "app.assessment.processing.extract_text_from_response_dict",
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
            "app.assessment.processing.extract_text_from_response_dict",
            return_value="",
        ):
            raw = [{"key": "row_0", "response": {"candidates": []}, "error": None}]
            results = parse_assessment_output(raw, "google")
        assert results[0]["output"] is None
        assert results[0]["error"] == "Empty response output"

    def test_google_native_provider_accepted(self) -> None:
        from unittest.mock import patch

        with patch(
            "app.assessment.processing.extract_text_from_response_dict",
            return_value="out",
        ):
            raw = [{"key": "row_0", "response": {"x": 1}, "error": None}]
            results = parse_assessment_output(raw, "google-native")
        assert results[0]["output"] == "out"


class TestGetBatchProvider:
    def test_unsupported_provider_raises(self) -> None:
        session = MagicMock()
        with pytest.raises(ValueError, match="Unsupported provider"):
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
            "app.assessment.processing.get_openai_client", return_value=mock_client
        ), patch("app.assessment.processing.OpenAIBatchProvider") as mock_cls:
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
        with patch("app.assessment.processing.GeminiClient") as mock_cls, patch(
            "app.assessment.processing.GeminiBatchProvider"
        ) as mock_batch_cls:
            mock_cls.from_credentials.return_value = mock_gemini
            _get_batch_provider(
                session=session,
                provider_name="google",
                organization_id=1,
                project_id=1,
            )
        mock_batch_cls.assert_called_once_with(client=mock_gemini.client)


class TestPollAllPendingAssessments:
    @pytest.mark.asyncio
    async def test_delegates_to_cron(self) -> None:
        session = MagicMock()
        expected = {"processed": 2, "failed": 0}
        with patch(
            "app.assessment.cron.poll_all_pending_assessment_evaluations",
            new=AsyncMock(return_value=expected),
        ):
            result = await poll_all_pending_assessments(session=session)
        assert result == expected
