"""Unit tests for the native LLM-as-judge scoring primitives (`crud/evaluations/judge.py`).

Covers the ground-truth and adherence-to-prompt judge slices of the three-metric
SRD: FR-2 (score in [0,1] + reasoning), FR-3 (ground truth judged against the
golden answer), FR-9 (zero-config default prompt), FR-15 (malformed → raises so
the row isolates), plus the run-level gating that drops a metric whose run input
could not be resolved. The single external boundary — the OpenAI judge completion
— is mocked at `_create_judge_response`; parsing/prompt-composition helpers are pure.
"""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import openai
import pytest
from sqlmodel import Session

from app.core.config import settings
from app.crud.evaluations.judge import (
    JUDGE_COST_STAGE,
    METRIC_REGISTRY,
    JudgeInputEnum,
    JudgeMetricEnum,
    MetricScore,
    _applicable_metrics,
    _compose_judge_input,
    _compose_system_prompt,
    _parse_judge_output,
    build_judge_params,
    enabled_metric_specs,
    judge_row,
)
from app.crud.evaluations.score import (
    GROUND_TRUTH_JUDGE_PROMPT,
    GROUND_TRUTH_SCORE_NAME,
    JUDGE_SYSTEM_PREAMBLE,
    KNOWLEDGE_BASE_JUDGE_PROMPT,
    KNOWLEDGE_BASE_SCORE_NAME,
    PROMPT_SCORE_NAME,
)

_GROUND_TRUTH_SPEC = METRIC_REGISTRY[JudgeMetricEnum.GROUND_TRUTH]
_KNOWLEDGE_BASE_SPEC = METRIC_REGISTRY[JudgeMetricEnum.KNOWLEDGE_BASE]


def _judge_response(payload: dict | str, *, usage=(15, 8, 23)):
    """Mimic the OpenAI Responses SDK object the judge parses (`output_text` + usage)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    input_tokens, output_tokens, total_tokens = usage
    return SimpleNamespace(
        output_text=text,
        output=[],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        ),
    )


def _both_metric_specs():
    """The metric set a run gets once its assistant prompt resolves."""
    return enabled_metric_specs(
        available_run_inputs=frozenset({JudgeInputEnum.CONFIG_PROMPT})
    )


class TestParseJudgeOutput:
    """FR-2 / FR-15: well-formed replies parse; malformed ones raise."""

    def test_parses_well_formed_ground_truth(self) -> None:
        specs = enabled_metric_specs()
        result = _parse_judge_output(
            json.dumps(
                {"ground_truth": {"score": 0.75, "reasoning": "close paraphrase"}}
            ),
            specs,
        )
        assert set(result) == {JudgeMetricEnum.GROUND_TRUTH}
        score = result[JudgeMetricEnum.GROUND_TRUTH]
        assert score.score == 0.75
        assert score.reasoning == "close paraphrase"

    def test_extracts_json_from_prose_wrapper(self) -> None:
        specs = enabled_metric_specs()
        wrapped = (
            'Here is my grade:\n{"ground_truth": {"score": 0.4, "reasoning": '
            '"missing a key fact"}}\nThanks.'
        )
        result = _parse_judge_output(wrapped, specs)
        assert result[JudgeMetricEnum.GROUND_TRUTH].score == 0.4

    def test_drops_only_the_missing_metric_when_others_present(self) -> None:
        # Grade against two specs (ground_truth + a synthetic sibling); a reply that
        # scores only ground_truth must drop the sibling from the map, not raise.
        gt_spec = enabled_metric_specs()[0]
        sibling_key = SimpleNamespace(value="sibling_metric")
        sibling = replace(gt_spec, key=sibling_key)

        result = _parse_judge_output(
            json.dumps({"ground_truth": {"score": 0.9, "reasoning": "correct"}}),
            [gt_spec, sibling],
        )
        assert list(result) == [JudgeMetricEnum.GROUND_TRUTH]

    def test_parses_both_metrics_when_both_requested(self) -> None:
        result = _parse_judge_output(
            json.dumps(
                {
                    "ground_truth": {"score": 0.8, "reasoning": "same facts"},
                    "knowledge_base": {
                        "score": 0.6,
                        "reasoning": "one unsupported claim",
                    },
                }
            ),
            [_GROUND_TRUTH_SPEC, _KNOWLEDGE_BASE_SPEC],
        )
        assert set(result) == {
            JudgeMetricEnum.GROUND_TRUTH,
            JudgeMetricEnum.KNOWLEDGE_BASE,
        }
        assert result[JudgeMetricEnum.KNOWLEDGE_BASE].score == 0.6

    def test_missing_knowledge_base_leaves_only_ground_truth(self) -> None:
        # Both metrics requested but the reply omits knowledge_base: that metric is
        # silently unscoreable for the row — no raise, ground_truth still parsed.
        result = _parse_judge_output(
            json.dumps({"ground_truth": {"score": 0.9, "reasoning": "correct"}}),
            [_GROUND_TRUTH_SPEC, _KNOWLEDGE_BASE_SPEC],
        )
        assert set(result) == {JudgeMetricEnum.GROUND_TRUTH}

    def test_raises_when_no_enabled_metric_scored(self) -> None:
        specs = enabled_metric_specs()
        with pytest.raises(ValueError, match="scored no enabled metric"):
            _parse_judge_output(json.dumps({"unrelated": {"score": 0.5}}), specs)

    def test_raises_on_empty_response(self) -> None:
        with pytest.raises(ValueError, match="empty judge response"):
            _parse_judge_output("   ", enabled_metric_specs())

    def test_raises_on_non_json(self) -> None:
        with pytest.raises(ValueError, match="no JSON object"):
            _parse_judge_output("the answer is basically fine", enabled_metric_specs())

    def test_raises_on_score_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of .0, 1."):
            _parse_judge_output(
                json.dumps({"ground_truth": {"score": 1.4, "reasoning": "x"}}),
                enabled_metric_specs(),
            )

    def test_raises_on_empty_reasoning(self) -> None:
        with pytest.raises(ValueError, match="empty 'reasoning'"):
            _parse_judge_output(
                json.dumps({"ground_truth": {"score": 0.8, "reasoning": "  "}}),
                enabled_metric_specs(),
            )

    def test_raises_on_non_numeric_score(self) -> None:
        with pytest.raises(ValueError, match="not a number"):
            _parse_judge_output(
                json.dumps({"ground_truth": {"score": "high", "reasoning": "x"}}),
                enabled_metric_specs(),
            )


class TestComposeJudgeInput:
    """FR-3: the ground-truth judge input carries question + answer + golden answer."""

    def test_input_contains_all_three_ground_truth_inputs(self) -> None:
        composed = _compose_judge_input(
            metrics=enabled_metric_specs(),
            inputs={
                JudgeInputEnum.QUESTION: "What is the capital of France?",
                JudgeInputEnum.GENERATED_ANSWER: "Paris is the capital.",
                JudgeInputEnum.GOLDEN_ANSWER: "Paris",
            },
        )
        assert "What is the capital of France?" in composed
        assert "Paris is the capital." in composed
        assert "Golden (reference) answer:\nParis" in composed
        # The output contract names the metric key the judge must return.
        assert "ground_truth" in composed

    def test_config_prompt_block_renders_first_and_labelled(self) -> None:
        composed = _compose_judge_input(
            metrics=_both_metric_specs(),
            inputs={
                JudgeInputEnum.CONFIG_PROMPT: "Only answer in Hindi.",
                JudgeInputEnum.QUESTION: "What is the capital of France?",
                JudgeInputEnum.GENERATED_ANSWER: "Paris is the capital.",
                JudgeInputEnum.GOLDEN_ANSWER: "Paris",
            },
        )
        assert composed.startswith(
            "Assistant's configured instructions:\nOnly answer in Hindi."
        )
        assert composed.index("Question:") < composed.index("Generated answer:")
        assert "ground_truth, prompt" in composed


class TestApplicableMetrics:
    """A metric runs on a row only when all its required inputs are present + non-empty."""

    def test_no_chunks_yields_ground_truth_only(self) -> None:
        applicable = _applicable_metrics(
            enabled_metric_specs(),
            {
                JudgeInputEnum.QUESTION: "Q",
                JudgeInputEnum.GENERATED_ANSWER: "A",
                JudgeInputEnum.GOLDEN_ANSWER: "G",
                JudgeInputEnum.RETRIEVED_CHUNKS: "",
            },
        )
        assert [s.key for s in applicable] == [JudgeMetricEnum.GROUND_TRUTH]

    def test_chunks_present_yields_both_metrics(self) -> None:
        applicable = _applicable_metrics(
            enabled_metric_specs(),
            {
                JudgeInputEnum.QUESTION: "Q",
                JudgeInputEnum.GENERATED_ANSWER: "A",
                JudgeInputEnum.GOLDEN_ANSWER: "G",
                JudgeInputEnum.RETRIEVED_CHUNKS: "chunk text",
            },
        )
        assert [s.key for s in applicable] == [
            JudgeMetricEnum.GROUND_TRUTH,
            JudgeMetricEnum.KNOWLEDGE_BASE,
        ]

    def test_empty_question_drops_ground_truth(self) -> None:
        applicable = _applicable_metrics(
            enabled_metric_specs(),
            {
                JudgeInputEnum.QUESTION: "",
                JudgeInputEnum.GENERATED_ANSWER: "A",
                JudgeInputEnum.GOLDEN_ANSWER: "G",
                JudgeInputEnum.RETRIEVED_CHUNKS: "chunk text",
            },
        )
        # QUESTION is empty, so ground_truth is inapplicable; knowledge_base (no QUESTION
        # requirement) still applies from the generated answer + chunks.
        assert JudgeMetricEnum.GROUND_TRUTH not in [s.key for s in applicable]
        assert [s.key for s in applicable] == [JudgeMetricEnum.KNOWLEDGE_BASE]


class TestBuildJudgeParams:
    """FR-9: system-config judging uses the fallback model + built-in ground-truth prompt.

    The judge is a reasoning model (gpt-5-mini) that rejects a custom temperature, so
    the request never carries one.
    """

    def test_defaults_to_fallback_model_and_builtin_prompt(self, db: Session) -> None:
        base_params = build_judge_params(session=db)

        assert base_params["model"] == settings.EVAL_JUDGE_MODEL
        assert base_params["model"] == "gpt-5-mini"
        assert "temperature" not in base_params
        # Instructions are the per-row applicable-metric subset, so they are composed
        # in judge_row, never baked into the shared base params.
        assert "instructions" not in base_params

        system_prompt = _compose_system_prompt(enabled_metric_specs())
        assert JUDGE_SYSTEM_PREAMBLE in system_prompt
        assert GROUND_TRUTH_JUDGE_PROMPT in system_prompt


class TestJudgeRow:
    """FR-3 / FR-15: one combined call per row, returning scores + usage; raises isolate."""

    def _base_params(self, db: Session) -> dict:
        return build_judge_params(session=db)

    @staticmethod
    def _gt_inputs(
        question: str = "Q", generated: str = "A", golden: str = "G"
    ) -> dict[JudgeInputEnum, str]:
        return {
            JudgeInputEnum.QUESTION: question,
            JudgeInputEnum.GENERATED_ANSWER: generated,
            JudgeInputEnum.GOLDEN_ANSWER: golden,
        }

    def test_returns_metric_score_and_usage(self, db: Session) -> None:
        with patch(
            "app.crud.evaluations.judge._create_judge_response",
            return_value=_judge_response(
                {"ground_truth": {"score": 0.9, "reasoning": "same meaning"}}
            ),
        ):
            result = judge_row(
                openai_client=SimpleNamespace(),
                base_params=self._base_params(db),
                metrics=enabled_metric_specs(),
                inputs=self._gt_inputs(generated="A", golden="A-golden"),
            )

        assert result.metrics[JudgeMetricEnum.GROUND_TRUTH] == MetricScore(
            score=0.9, reasoning="same meaning"
        )
        assert result.usage == {
            "input_tokens": 15,
            "output_tokens": 8,
            "total_tokens": 23,
        }

    def test_chunkless_row_judges_ground_truth_only(self, db: Session) -> None:
        captured: dict = {}

        def _capture(_client, params):
            captured.update(params)
            return _judge_response({"ground_truth": {"score": 0.9, "reasoning": "ok"}})

        with patch(
            "app.crud.evaluations.judge._create_judge_response", side_effect=_capture
        ):
            result = judge_row(
                openai_client=SimpleNamespace(),
                base_params=self._base_params(db),
                metrics=enabled_metric_specs(),
                inputs=self._gt_inputs(),
            )

        assert set(result.metrics) == {JudgeMetricEnum.GROUND_TRUTH}
        # With no chunks, knowledge_base is dropped: its rubric, chunk label, and key
        # must never reach the judge call for this row.
        assert KNOWLEDGE_BASE_JUDGE_PROMPT not in captured["instructions"]
        assert "Retrieved knowledge-base chunks" not in captured["input"]
        assert "knowledge_base" not in captured["input"]

    def test_chunk_present_row_judges_both_metrics(self, db: Session) -> None:
        with patch(
            "app.crud.evaluations.judge._create_judge_response",
            return_value=_judge_response(
                {
                    "ground_truth": {"score": 0.8, "reasoning": "gt"},
                    "knowledge_base": {"score": 0.7, "reasoning": "kb"},
                }
            ),
        ):
            result = judge_row(
                openai_client=SimpleNamespace(),
                base_params=self._base_params(db),
                metrics=enabled_metric_specs(),
                inputs={
                    **self._gt_inputs(),
                    JudgeInputEnum.RETRIEVED_CHUNKS: "supporting chunk",
                },
            )

        assert set(result.metrics) == {
            JudgeMetricEnum.GROUND_TRUTH,
            JudgeMetricEnum.KNOWLEDGE_BASE,
        }
        assert result.metrics[JudgeMetricEnum.KNOWLEDGE_BASE].score == 0.7

    def test_chunkless_prompt_is_byte_identical_to_ground_truth_only(
        self, db: Session
    ) -> None:
        """Adding knowledge_base to the registry leaves the chunk-less path unchanged."""
        captured: dict = {}

        def _capture(_client, params):
            captured.update(params)
            return _judge_response({"ground_truth": {"score": 0.5, "reasoning": "x"}})

        with patch(
            "app.crud.evaluations.judge._create_judge_response", side_effect=_capture
        ):
            judge_row(
                openai_client=SimpleNamespace(),
                base_params=self._base_params(db),
                metrics=enabled_metric_specs(),
                inputs=self._gt_inputs(question="Qx", generated="Ax", golden="Gx"),
            )

        assert captured["instructions"] == _compose_system_prompt([_GROUND_TRUTH_SPEC])
        assert "Question:\nQx" in captured["input"]
        assert "Generated answer:\nAx" in captured["input"]
        assert "Golden (reference) answer:\nGx" in captured["input"]
        assert "Include exactly these metric keys: ground_truth." in captured["input"]

    def test_judge_call_receives_question_answer_and_golden(self, db: Session) -> None:
        """FR-3: the row's golden answer reaches the judge completion input."""
        captured: dict = {}

        def _capture(_client, params):
            captured.update(params)
            return _judge_response(
                {"ground_truth": {"score": 0.5, "reasoning": "partial"}}
            )

        with patch(
            "app.crud.evaluations.judge._create_judge_response", side_effect=_capture
        ):
            judge_row(
                openai_client=SimpleNamespace(),
                base_params=self._base_params(db),
                metrics=enabled_metric_specs(),
                inputs=self._gt_inputs(
                    question="Who wrote Hamlet?",
                    generated="Shakespeare wrote it.",
                    golden="William Shakespeare",
                ),
            )

        assert "Who wrote Hamlet?" in captured["input"]
        assert "Shakespeare wrote it." in captured["input"]
        assert "William Shakespeare" in captured["input"]

    def test_malformed_output_raises_for_row_isolation(self, db: Session) -> None:
        with patch(
            "app.crud.evaluations.judge._create_judge_response",
            return_value=_judge_response("not json at all"),
        ):
            with pytest.raises(ValueError):
                judge_row(
                    openai_client=SimpleNamespace(),
                    base_params=self._base_params(db),
                    metrics=enabled_metric_specs(),
                    inputs=self._gt_inputs(),
                )

    def test_openai_error_propagates_for_row_isolation(self, db: Session) -> None:
        with patch(
            "app.crud.evaluations.judge._create_judge_response",
            side_effect=openai.OpenAIError("judge provider down"),
        ):
            with pytest.raises(openai.OpenAIError):
                judge_row(
                    openai_client=SimpleNamespace(),
                    base_params=self._base_params(db),
                    metrics=enabled_metric_specs(),
                    inputs=self._gt_inputs(),
                )


class TestEnabledMetricSpecs:
    """Run-level input gating: a metric needing an unresolved run input is dropped.

    knowledge_base has no run-level input (its chunks are per-row), so it is always
    enabled at run level and only drops per row via `_applicable_metrics`. prompt
    needs the run-level config prompt, so it is gated here.
    """

    def test_without_run_inputs_ground_truth_and_knowledge_base_enabled(self) -> None:
        specs = enabled_metric_specs()
        assert [s.key for s in specs] == [
            JudgeMetricEnum.GROUND_TRUTH,
            JudgeMetricEnum.KNOWLEDGE_BASE,
        ]
        assert _GROUND_TRUTH_SPEC.score_name == GROUND_TRUTH_SCORE_NAME
        assert JudgeInputEnum.CONFIG_PROMPT not in _GROUND_TRUTH_SPEC.required_inputs
        assert _KNOWLEDGE_BASE_SPEC.score_name == KNOWLEDGE_BASE_SCORE_NAME

    def test_config_prompt_available_enables_all_three_metrics(self) -> None:
        specs = enabled_metric_specs(
            available_run_inputs=frozenset({JudgeInputEnum.CONFIG_PROMPT})
        )
        assert [s.key for s in specs] == [
            JudgeMetricEnum.GROUND_TRUTH,
            JudgeMetricEnum.PROMPT,
            JudgeMetricEnum.KNOWLEDGE_BASE,
        ]
        prompt_spec = specs[1]
        assert prompt_spec.score_name == PROMPT_SCORE_NAME
        assert JudgeInputEnum.CONFIG_PROMPT in prompt_spec.required_inputs

    def test_all_metrics_share_one_judge_cost_stage(self) -> None:
        # One combined call per row, so tokens can't be split per metric.
        assert JUDGE_COST_STAGE == "judge"
