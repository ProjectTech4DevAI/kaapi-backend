"""Cosine scoring extracted from Stage 3 (`fast_cosine.py`).

Covers the ref-keying rule (trace_id when traced, item_id when not) and the
unscoreable classification, since both decide what the UI can show for a row.
"""

from typing import Any

from app.crud.evaluations.fast_cosine import (
    build_item_refs,
    classify_empty_side,
    score_cosine_run,
)
from app.crud.evaluations.score import (
    COSINE_SCORE_NAME,
    UNSCOREABLE_EMBEDDING_FAILED,
    UNSCOREABLE_EMPTY_GROUND_TRUTH,
    UNSCOREABLE_EMPTY_OUTPUT,
)


def _response(
    item_id: str, output: str = "out", ground_truth: str = "gt"
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "question": "q",
        "generated_output": output,
        "ground_truth": ground_truth,
    }


def _embedding(item_id: str, vector: list[float], other: list[float]) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "output_embedding": vector,
        "ground_truth_embedding": other,
        "failed": False,
    }


class TestClassifyEmptySide:
    def test_empty_output_wins_over_empty_ground_truth(self) -> None:
        response = _response("i", output="", ground_truth="")
        assert classify_empty_side(response) == UNSCOREABLE_EMPTY_OUTPUT

    def test_empty_ground_truth_alone(self) -> None:
        response = _response("i", ground_truth="")
        assert classify_empty_side(response) == UNSCOREABLE_EMPTY_GROUND_TRUTH

    def test_both_sides_present_is_none(self) -> None:
        assert classify_empty_side(_response("i")) is None


class TestBuildItemRefs:
    def test_prefers_the_trace_id_when_the_run_is_traced(self) -> None:
        refs = build_item_refs([_response("a"), _response("b")], {"a": "trace-a"})
        assert refs == {"a": "trace-a", "b": "b"}

    def test_falls_back_to_item_id_when_untraced(self) -> None:
        refs = build_item_refs([_response("a")], {})
        assert refs == {"a": "a"}


class TestScoreCosineRun:
    def test_identical_vectors_score_one_and_land_in_the_summary(self) -> None:
        responses = [_response("a")]
        item_refs = build_item_refs(responses, {})
        result = score_cosine_run(
            response_results=responses,
            embedding_results=[_embedding("a", [1.0, 0.0], [1.0, 0.0])],
            item_refs=item_refs,
            trace_id_mapping={},
            total_items=1,
        )
        assert result.item_id_to_score["a"] == 1.0
        assert result.per_item_scores == {"a": 1.0}
        assert result.unscoreable == {}

        summary = next(
            s for s in result.summary_scores if s["name"] == COSINE_SCORE_NAME
        )
        assert summary["avg"] == 1.0
        assert summary["total_pairs"] == 1

    def test_missing_embedding_pair_is_unscoreable_not_zero(self) -> None:
        responses = [_response("a")]
        item_refs = build_item_refs(responses, {})
        result = score_cosine_run(
            response_results=responses,
            embedding_results=[],
            item_refs=item_refs,
            trace_id_mapping={},
            total_items=1,
        )
        assert result.unscoreable == {"a": UNSCOREABLE_EMBEDDING_FAILED}
        assert result.item_id_to_score == {}
        summary = next(
            s for s in result.summary_scores if s["name"] == COSINE_SCORE_NAME
        )
        assert summary["total_pairs"] == 0

    def test_empty_side_beats_embedding_failed_as_the_reason(self) -> None:
        responses = [_response("a", output="")]
        result = score_cosine_run(
            response_results=responses,
            embedding_results=[],
            item_refs=build_item_refs(responses, {}),
            trace_id_mapping={},
            total_items=1,
        )
        assert result.unscoreable == {"a": UNSCOREABLE_EMPTY_OUTPUT}

    def test_untraced_run_writes_nothing_back_to_langfuse(self) -> None:
        responses = [_response("a")]
        result = score_cosine_run(
            response_results=responses,
            embedding_results=[_embedding("a", [1.0, 0.0], [1.0, 0.0])],
            item_refs=build_item_refs(responses, {}),
            trace_id_mapping={},
            total_items=1,
        )
        assert result.write_items == []

    def test_traced_run_writes_scores_and_unscoreable_reasons(self) -> None:
        responses = [_response("a"), _response("b", output="")]
        trace_id_mapping = {"a": "trace-a", "b": "trace-b"}
        result = score_cosine_run(
            response_results=responses,
            embedding_results=[_embedding("a", [1.0, 0.0], [1.0, 0.0])],
            item_refs=build_item_refs(responses, trace_id_mapping),
            trace_id_mapping=trace_id_mapping,
            total_items=2,
        )
        by_trace = {w["trace_id"]: w for w in result.write_items}
        assert by_trace["trace-a"]["cosine_similarity"] == 1.0
        assert by_trace["trace-b"]["unscoreable"] is True
        assert by_trace["trace-b"]["reason"] == UNSCOREABLE_EMPTY_OUTPUT

    def test_per_item_scores_are_keyed_by_ref_not_item_id(self) -> None:
        responses = [_response("a")]
        trace_id_mapping = {"a": "trace-a"}
        result = score_cosine_run(
            response_results=responses,
            embedding_results=[_embedding("a", [1.0, 0.0], [1.0, 0.0])],
            item_refs=build_item_refs(responses, trace_id_mapping),
            trace_id_mapping=trace_id_mapping,
            total_items=1,
        )
        assert result.per_item_scores == {"trace-a": 1.0}
