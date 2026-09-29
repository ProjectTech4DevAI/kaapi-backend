"""Run-level judge helpers extracted from Stage 3 (`judge_stage.py`).

The per-row judge call itself is covered by `test_judge.py`; these cover the
run-level wrapping — which rows are judgeable, how a row's input blocks are
assembled, and the metric rollup a run summary is built from.
"""

from app.crud.evaluations.judge import (
    METRIC_REGISTRY,
    JudgeInputEnum,
    JudgeMetricEnum,
    JudgeResult,
    MetricScore,
)
from app.crud.evaluations.judge_stage import (
    build_judge_inputs,
    build_metric_summary_scores,
    select_judgeable_rows,
)
from app.crud.evaluations.score import GROUND_TRUTH_SCORE_NAME, PROMPT_SCORE_NAME

METRICS = list(METRIC_REGISTRY.values())


def _judge_result(**metrics: MetricScore) -> JudgeResult:
    return JudgeResult(
        metrics={JudgeMetricEnum(k): v for k, v in metrics.items()},
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


class TestSelectJudgeableRows:
    def test_keeps_rows_with_both_sides(self) -> None:
        rows = [{"item_id": "a", "generated_output": "o", "ground_truth": "g"}]
        assert select_judgeable_rows(rows) == rows

    def test_drops_rows_missing_either_side(self) -> None:
        rows = [
            {"item_id": "a", "generated_output": "", "ground_truth": "g"},
            {"item_id": "b", "generated_output": "o", "ground_truth": ""},
            {"item_id": "c", "generated_output": "o", "ground_truth": "g"},
        ]
        assert [r["item_id"] for r in select_judgeable_rows(rows)] == ["c"]


class TestBuildJudgeInputs:
    def test_maps_the_response_onto_the_input_blocks(self) -> None:
        inputs = build_judge_inputs(
            response={
                "question": "q",
                "generated_output": "out",
                "ground_truth": "gt",
                "retrieved_chunks": [{"text": "one"}, {"text": "two"}],
            },
            config_prompt="be nice",
        )
        assert inputs[JudgeInputEnum.CONFIG_PROMPT] == "be nice"
        assert inputs[JudgeInputEnum.QUESTION] == "q"
        assert inputs[JudgeInputEnum.GENERATED_ANSWER] == "out"
        assert inputs[JudgeInputEnum.GOLDEN_ANSWER] == "gt"
        assert inputs[JudgeInputEnum.RETRIEVED_CHUNKS] == "one\n---\ntwo"

    def test_chunks_without_text_are_skipped(self) -> None:
        inputs = build_judge_inputs(
            response={
                "retrieved_chunks": [{"text": ""}, {"score": 1}, {"text": "kept"}]
            },
            config_prompt="",
        )
        assert inputs[JudgeInputEnum.RETRIEVED_CHUNKS] == "kept"

    def test_absent_fields_become_empty_blocks(self) -> None:
        inputs = build_judge_inputs(response={}, config_prompt="")
        assert all(value == "" for value in inputs.values())


class TestBuildMetricSummaryScores:
    def test_averages_each_metric_over_the_rows_that_scored_it(self) -> None:
        judge_results = {
            "a": _judge_result(ground_truth=MetricScore(score=4, reasoning="r")),
            "b": _judge_result(ground_truth=MetricScore(score=2, reasoning="r")),
        }
        summaries = build_metric_summary_scores(
            metrics=METRICS, judge_results=judge_results
        )
        ground_truth = next(
            s for s in summaries if s["name"] == GROUND_TRUTH_SCORE_NAME
        )
        assert ground_truth["avg"] == 3.0
        assert ground_truth["std"] == 1.0
        assert ground_truth["total_pairs"] == 2
        assert ground_truth["data_type"] == "NUMERIC"

    def test_a_metric_no_row_scored_is_omitted_entirely(self) -> None:
        judge_results = {
            "a": _judge_result(ground_truth=MetricScore(score=5, reasoning="r"))
        }
        summaries = build_metric_summary_scores(
            metrics=METRICS, judge_results=judge_results
        )
        assert {s["name"] for s in summaries} == {GROUND_TRUTH_SCORE_NAME}

    def test_only_the_rows_that_scored_count_toward_total_pairs(self) -> None:
        judge_results = {
            "a": _judge_result(
                ground_truth=MetricScore(score=5, reasoning="r"),
                prompt=MetricScore(score=3, reasoning="r"),
            ),
            "b": _judge_result(ground_truth=MetricScore(score=5, reasoning="r")),
        }
        summaries = {
            s["name"]: s
            for s in build_metric_summary_scores(
                metrics=METRICS, judge_results=judge_results
            )
        }
        assert summaries[GROUND_TRUTH_SCORE_NAME]["total_pairs"] == 2
        assert summaries[PROMPT_SCORE_NAME]["total_pairs"] == 1

    def test_no_judge_results_yields_no_summaries(self) -> None:
        assert build_metric_summary_scores(metrics=METRICS, judge_results={}) == []
