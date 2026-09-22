"""Trace-record building extracted from Stage 3 (`fast_traces.py`).

The trace unit is the v2 source of truth for per-row judge scores, so the shape
built here (and the KB placeholder wording) is what the read path serves.
"""

from typing import Any

from app.crud.evaluations.fast_traces import build_trace_records, format_top_kb_matches
from app.crud.evaluations.judge import (
    METRIC_REGISTRY,
    JudgeMetricEnum,
    JudgeResult,
    MetricScore,
)
from app.crud.evaluations.score import (
    COSINE_SCORE_NAME,
    DEFAULT_CATEGORY,
    GROUND_TRUTH_SCORE_NAME,
    JUDGE_FAILED_REASON,
    KNOWLEDGE_BASE_SCORE_NAME,
    UNSCOREABLE_EMPTY_OUTPUT,
    VerdictEnum,
)

METRICS = list(METRIC_REGISTRY.values())


def _response(item_id: str, **overrides: Any) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "question": "q",
        "generated_output": "out",
        "ground_truth": "gt",
        "question_id": 1,
        **overrides,
    }


def _scores_by_name(trace: dict[str, Any]) -> dict[str, Any]:
    return {s["name"]: s for s in trace["scores"]}


class TestFormatTopKbMatches:
    def test_renders_filename_and_percentage(self) -> None:
        chunks = [{"filename": "biu-1.pdf", "score": 0.906}]
        assert format_top_kb_matches(chunks) == "biu-1.pdf (90.6%)"

    def test_caps_at_three_matches(self) -> None:
        chunks = [{"filename": f"f{i}.pdf", "score": 0.5} for i in range(5)]
        assert len(format_top_kb_matches(chunks).split(", ")) == 3

    def test_missing_filename_renders_unknown(self) -> None:
        assert format_top_kb_matches([{"score": 0.5}]) == "unknown (50.0%)"

    def test_no_chunks_is_empty_string(self) -> None:
        assert format_top_kb_matches([]) == ""


class TestBuildTraceRecordsCosine:
    def test_scored_row_carries_the_cosine_entry(self) -> None:
        traces = build_trace_records(
            response_results=[_response("a")],
            item_refs={"a": "a"},
            is_judge_run=False,
            judge_results={},
            metrics=[],
            cosine_by_item_id={"a": 0.912},
            unscoreable={},
        )
        score = _scores_by_name(traces[0])[COSINE_SCORE_NAME]
        assert score["value"] == 0.91
        assert score.get("unscoreable") is None

    def test_unscoreable_row_gets_a_zero_placeholder(self) -> None:
        traces = build_trace_records(
            response_results=[_response("a", generated_output="")],
            item_refs={"a": "a"},
            is_judge_run=False,
            judge_results={},
            metrics=[],
            cosine_by_item_id={},
            unscoreable={"a": UNSCOREABLE_EMPTY_OUTPUT},
        )
        score = _scores_by_name(traces[0])[COSINE_SCORE_NAME]
        assert score["value"] == 0
        assert score["unscoreable"] is True
        assert UNSCOREABLE_EMPTY_OUTPUT in score["comment"]

    def test_judge_failed_reason_gets_no_cosine_placeholder(self) -> None:
        traces = build_trace_records(
            response_results=[_response("a")],
            item_refs={"a": "a"},
            is_judge_run=False,
            judge_results={},
            metrics=[],
            cosine_by_item_id={},
            unscoreable={"a": JUDGE_FAILED_REASON},
        )
        assert traces[0]["scores"] == []


class TestBuildTraceRecordsJudge:
    @staticmethod
    def _judge_result(**metrics: MetricScore) -> JudgeResult:
        return JudgeResult(
            metrics={JudgeMetricEnum(k): v for k, v in metrics.items()},
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    def test_each_metric_carries_score_reasoning_and_verdict(self) -> None:
        traces = build_trace_records(
            response_results=[_response("a")],
            item_refs={"a": "a"},
            is_judge_run=True,
            judge_results={
                "a": self._judge_result(
                    ground_truth=MetricScore(score=5, reasoning="spot on")
                )
            },
            metrics=METRICS,
            cosine_by_item_id={},
            unscoreable={},
        )
        score = _scores_by_name(traces[0])[GROUND_TRUTH_SCORE_NAME]
        assert score["value"] == 5
        assert score["comment"] == "spot on"
        assert score["verdict"] is VerdictEnum.GOOD

    def test_judge_run_never_carries_a_cosine_entry(self) -> None:
        traces = build_trace_records(
            response_results=[_response("a")],
            item_refs={"a": "a"},
            is_judge_run=True,
            judge_results={
                "a": self._judge_result(
                    ground_truth=MetricScore(score=3, reasoning="partly")
                )
            },
            metrics=METRICS,
            cosine_by_item_id={"a": 0.5},
            unscoreable={},
        )
        assert COSINE_SCORE_NAME not in _scores_by_name(traces[0])

    def test_kb_metric_appends_top_matches_to_its_reasoning(self) -> None:
        traces = build_trace_records(
            response_results=[
                _response(
                    "a",
                    retrieved_chunks=[
                        {"filename": "faq.pdf", "score": 0.8, "text": "t"}
                    ],
                )
            ],
            item_refs={"a": "a"},
            is_judge_run=True,
            judge_results={
                "a": self._judge_result(
                    knowledge_base=MetricScore(score=4, reasoning="grounded")
                )
            },
            metrics=METRICS,
            cosine_by_item_id={},
            unscoreable={},
        )
        comment = _scores_by_name(traces[0])[KNOWLEDGE_BASE_SCORE_NAME]["comment"]
        assert comment == "grounded | Top matches: faq.pdf (80.0%)"

    def test_kb_dropped_with_no_chunks_reads_as_not_queried(self) -> None:
        traces = build_trace_records(
            response_results=[_response("a")],
            item_refs={"a": "a"},
            is_judge_run=True,
            judge_results={
                "a": self._judge_result(
                    ground_truth=MetricScore(score=4, reasoning="ok")
                )
            },
            metrics=METRICS,
            cosine_by_item_id={},
            unscoreable={},
        )
        kb = _scores_by_name(traces[0])[KNOWLEDGE_BASE_SCORE_NAME]
        assert kb["value"] == "N/A"
        assert kb["unscoreable"] is True
        assert kb["comment"] == "Knowledge base not queried."

    def test_kb_dropped_despite_chunks_reads_as_unavailable(self) -> None:
        traces = build_trace_records(
            response_results=[
                _response(
                    "a",
                    retrieved_chunks=[{"filename": "f.pdf", "score": 0.4, "text": "t"}],
                )
            ],
            item_refs={"a": "a"},
            is_judge_run=True,
            judge_results={
                "a": self._judge_result(
                    ground_truth=MetricScore(score=4, reasoning="ok")
                )
            },
            metrics=METRICS,
            cosine_by_item_id={},
            unscoreable={},
        )
        kb = _scores_by_name(traces[0])[KNOWLEDGE_BASE_SCORE_NAME]
        assert kb["comment"] == "Knowledge base score unavailable for this row."

    def test_a_row_with_no_judge_result_still_gets_a_trace(self) -> None:
        traces = build_trace_records(
            response_results=[_response("a", generated_output="")],
            item_refs={"a": "a"},
            is_judge_run=True,
            judge_results={},
            metrics=METRICS,
            cosine_by_item_id={},
            unscoreable={"a": UNSCOREABLE_EMPTY_OUTPUT},
        )
        assert len(traces) == 1
        assert traces[0]["scores"] == []


class TestTraceEnvelope:
    def test_trace_is_keyed_by_ref_and_defaults_its_category(self) -> None:
        traces = build_trace_records(
            response_results=[_response("a")],
            item_refs={"a": "trace-a"},
            is_judge_run=False,
            judge_results={},
            metrics=[],
            cosine_by_item_id={},
            unscoreable={},
        )
        assert traces[0]["trace_id"] == "trace-a"
        assert traces[0]["category"] == DEFAULT_CATEGORY
        assert traces[0]["question_id"] == 1
        assert traces[0]["llm_answer"] == "out"
        assert traces[0]["ground_truth_answer"] == "gt"
