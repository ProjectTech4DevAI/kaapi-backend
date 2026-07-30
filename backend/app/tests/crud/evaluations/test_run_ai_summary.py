"""`generate_run_ai_summary` — the best-effort natural-language note on a run.

The summary reuses the judge's reasoning-model invocation: params built via
`map_kaapi_to_openai_params` (mocked here to skip its DB lookup) and one
`responses.create` call (the external boundary, also mocked). The Responses `input`
is a qualitative brief (band words + plain area names + consistency phrases + a
repetition line) — never raw scores or the internal "Adherence to X" labels. Every
failure mode (OpenAI error, generic error, empty output) must resolve to None
WITHOUT raising, leaving the deterministic overall to persist.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
import pytest

from app.crud.evaluations.score import OverallSummary
from app.crud.evaluations.summary import _consistency_read, generate_run_ai_summary

_MODEL = "gpt-5-mini"


def _overall() -> OverallSummary:
    return {
        "overall_score": 0.66,
        "verdict": "Good",
        "ai_summary": None,
        "breakdown": [
            {
                "name": "Adherence to Ground Truth",
                "key": "ground_truth",
                "score": 0.8,
                "weight": 0.5,
                "delta": 0.14,
                "verdict": "Good",
            },
            {
                "name": "Adherence to Knowledge Base",
                "key": "knowledge_base",
                "score": 0.6,
                "weight": 0.3,
                "delta": -0.06,
                "verdict": "Good",
            },
            {
                "name": "Adherence to Prompt",
                "key": "prompt",
                "score": 0.4,
                "weight": 0.2,
                "delta": -0.26,
                "verdict": "Needs Refinement",
            },
        ],
    }


def _summary_scores() -> list[dict]:
    # std per dimension drives the consistency read; names match the overall's dims.
    return [
        {"name": "Adherence to Ground Truth", "avg": 0.8, "std": 0.05},
        {"name": "Adherence to Knowledge Base", "avg": 0.6, "std": 0.15},
        {"name": "Adherence to Prompt", "avg": 0.4, "std": 0.34},
    ]


def _responses_result(output_text: str):
    return SimpleNamespace(output_text=output_text, output=[])


def _call(
    client: MagicMock,
    *,
    duplication_factor: int = 5,
    summary_scores: list[dict] | None = None,
) -> str | None:
    # The mapper does a real DB lookup (is_reasoning_model); patch it so the unit
    # test stays about the summary logic, not model resolution.
    with patch(
        "app.crud.evaluations.summary.map_kaapi_to_openai_params",
        return_value=({"model": _MODEL, "effort": "medium"}, []),
    ):
        return generate_run_ai_summary(
            session=MagicMock(),
            openai_client=client,
            model=_MODEL,
            overall=_overall(),
            run_name="run-x",
            summary_scores=summary_scores
            if summary_scores is not None
            else _summary_scores(),
            duplication_factor=duplication_factor,
        )


class TestHappyPath:
    def test_returns_the_responses_output_text_stripped(self) -> None:
        client = MagicMock()
        client.responses.create.return_value = _responses_result(
            "  Consistently grounded and on-tone across the set.  "
        )
        assert _call(client) == "Consistently grounded and on-tone across the set."


class TestQualitativeBrief:
    def _input_for(self, *, duplication_factor: int) -> str:
        client = MagicMock()
        client.responses.create.return_value = _responses_result("A note.")
        _call(client, duplication_factor=duplication_factor)
        return client.responses.create.call_args.kwargs["input"]

    def test_input_carries_plain_area_and_consistency_phrases_and_token_cap(
        self,
    ) -> None:
        client = MagicMock()
        client.responses.create.return_value = _responses_result("A note.")
        _call(client, duplication_factor=5)

        params = client.responses.create.call_args.kwargs
        brief = params["input"]
        assert params["max_output_tokens"] == 2000
        assert "temperature" not in params

        assert "Accuracy against the expected answers" in brief
        assert "Grounding in the source material" in brief
        assert "Tone and instruction-following" in brief

        assert "answers stayed consistent" in brief  # std 0.05
        assert "answers were mostly consistent, with some variation" in brief  # 0.15
        assert "answers varied" in brief  # 0.34

    def test_brief_leaks_no_raw_scores_or_internal_labels(self) -> None:
        brief = self._input_for(duplication_factor=5)
        assert "Adherence to" not in brief
        assert "verdict" not in brief
        # Raw score / weight / delta values must not cross into the brief.
        for leaked in ("0.66", "0.8", "0.5", "0.14", "-0.26"):
            assert leaked not in brief

    def test_repetition_line_states_the_repeat_count_when_gt_one(self) -> None:
        brief = self._input_for(duplication_factor=5)
        assert "Each question was asked 5 times." in brief

    def test_repetition_line_softens_when_asked_once(self) -> None:
        brief = self._input_for(duplication_factor=1)
        assert "asked once (no repetition to speak of)" in brief
        assert "Each question was asked 1 times" not in brief


class TestConsistencyRead:
    @pytest.mark.parametrize(
        ("std", "expected"),
        [
            (0.08, "answers stayed consistent"),
            (0.15, "answers were mostly consistent, with some variation"),
            (0.34, "answers varied"),
            (None, "consistency unknown"),
        ],
    )
    def test_bands_and_boundaries(self, std, expected) -> None:
        assert _consistency_read(std) == expected


class TestFailureIsNonFatal:
    def test_openai_error_returns_none_without_raising(self) -> None:
        client = MagicMock()
        client.responses.create.side_effect = openai.OpenAIError("provider down")
        assert _call(client) is None

    def test_generic_error_returns_none_without_raising(self) -> None:
        client = MagicMock()
        client.responses.create.side_effect = RuntimeError("unexpected shape")
        assert _call(client) is None

    @pytest.mark.parametrize("output_text", ["", "   \n\t"])
    def test_empty_or_whitespace_output_returns_none(self, output_text: str) -> None:
        client = MagicMock()
        client.responses.create.return_value = _responses_result(output_text)
        assert _call(client) is None

    def test_malformed_payload_parse_error_returns_none_without_raising(self) -> None:
        # extract_response_text runs inside the try, so a raise on an unexpected
        # Responses payload must degrade to None, never escape to fail the run.
        client = MagicMock()
        client.responses.create.return_value = (
            SimpleNamespace()
        )  # no output_text/output
        with patch(
            "app.crud.evaluations.summary.extract_response_text",
            side_effect=ValueError("unexpected Responses payload"),
        ):
            assert _call(client) is None
