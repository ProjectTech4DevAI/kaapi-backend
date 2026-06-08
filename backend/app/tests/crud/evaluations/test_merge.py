"""Tests for step-forward trace merge helpers used on Langfuse resync."""

from app.crud.evaluations.merge import (
    compute_summary_scores,
    merge_scores_step_forward,
    merge_trace_data,
)


def _trace(trace_id, value=1.0, name="accuracy", data_type="NUMERIC"):
    return {
        "trace_id": trace_id,
        "question": f"q{trace_id}",
        "llm_answer": "a",
        "ground_truth_answer": "g",
        "question_id": int(trace_id),
        "scores": [{"name": name, "value": value, "data_type": data_type}],
    }


class TestMergeTraceData:
    def test_empty_cache_returns_fresh(self):
        fresh = [_trace("0"), _trace("1")]
        merged, stats = merge_trace_data([], fresh)
        assert len(merged) == 2
        assert stats == {"reused": 0, "updated": 0, "added": 2}

    def test_grows_monotonically(self):
        """15 -> 24 -> 30 across successive syncs."""
        sync1 = [_trace(str(i)) for i in range(15)]
        sync2 = [_trace(str(i)) for i in range(5, 24)]
        sync3 = [_trace(str(i)) for i in range(30)]

        m1, _ = merge_trace_data([], sync1)
        m2, s2 = merge_trace_data(m1, sync2)
        m3, s3 = merge_trace_data(m2, sync3)

        assert len(m1) == 15
        assert len(m2) == 24
        assert s2 == {"reused": 15, "updated": 0, "added": 9}
        assert len(m3) == 30
        assert s3 == {"reused": 24, "updated": 0, "added": 6}

    def test_fewer_fresh_traces_keeps_cached(self):
        """A partial fetch (27) must not drop cached traces (29)."""
        cached = [_trace(str(i)) for i in range(29)]
        fresh = [_trace(str(i)) for i in range(27)]
        merged, stats = merge_trace_data(cached, fresh)
        assert len(merged) == 29
        assert stats["reused"] == 29
        assert stats["added"] == 0

    def test_additional_score_unions(self):
        """A trace gaining a new score is enriched, not replaced wholesale."""
        existing = [_trace("1", name="accuracy")]
        fresh = [
            {
                "trace_id": "1",
                "question": "q1",
                "llm_answer": "a",
                "ground_truth_answer": "g",
                "question_id": 1,
                "scores": [{"name": "relevance", "value": 0.9, "data_type": "NUMERIC"}],
            }
        ]
        merged, stats = merge_trace_data(existing, fresh)
        names = sorted(s["name"] for s in merged[0]["scores"])
        assert names == ["accuracy", "relevance"]
        assert stats["updated"] == 1

    def test_fresh_value_wins_on_conflict(self):
        existing = [_trace("1", value=0.5)]
        fresh = [_trace("1", value=0.8)]
        merged, _ = merge_trace_data(existing, fresh)
        assert merged[0]["scores"][0]["value"] == 0.8

    def test_identical_trace_is_reused(self):
        existing = [_trace("1", value=0.5)]
        fresh = [_trace("1", value=0.5)]
        merged, stats = merge_trace_data(existing, fresh)
        assert stats["reused"] == 1
        assert stats["updated"] == 0


class TestComputeSummaryScores:
    def test_numeric_summary(self):
        traces = [_trace("0", 0.6), _trace("1", 0.8)]
        summary = compute_summary_scores(traces)
        assert len(summary) == 1
        s = summary[0]
        assert s["name"] == "accuracy"
        assert s["total_pairs"] == 2
        assert s["avg"] == 0.7

    def test_categorical_distribution(self):
        traces = [
            _trace("0", "CORRECT", name="verdict", data_type="CATEGORICAL"),
            _trace("1", "CORRECT", name="verdict", data_type="CATEGORICAL"),
            _trace("2", "WRONG", name="verdict", data_type="CATEGORICAL"),
        ]
        summary = compute_summary_scores(traces)
        s = summary[0]
        assert s["distribution"] == {"CORRECT": 2, "WRONG": 1}
        assert s["total_pairs"] == 3

    def test_none_values_excluded(self):
        traces = [_trace("0", 1.0), _trace("1", None)]
        summary = compute_summary_scores(traces)
        assert summary[0]["total_pairs"] == 1


class TestMergeScoresStepForward:
    def test_summary_recomputed_from_union(self):
        """total_pairs reflects the merged union, not the (smaller) fresh fetch."""
        cached = {
            "summary_scores": [],
            "traces": [_trace(str(i)) for i in range(29)],
        }
        fresh = {
            "summary_scores": [],
            "traces": [_trace(str(i)) for i in range(27)],
        }
        merged, _ = merge_scores_step_forward(cached, fresh)
        assert len(merged["traces"]) == 29
        accuracy = next(s for s in merged["summary_scores"] if s["name"] == "accuracy")
        assert accuracy["total_pairs"] == 29

    def test_summary_only_score_preserved(self):
        """A summary metric with no per-trace value survives the merge."""
        cached = {
            "summary_scores": [{"name": "cosine_similarity", "avg": 0.91}],
            "traces": [_trace("0")],
        }
        fresh = {"summary_scores": [], "traces": [_trace("0")]}
        merged, _ = merge_scores_step_forward(cached, fresh)
        names = {s["name"] for s in merged["summary_scores"]}
        assert "cosine_similarity" in names
        assert "accuracy" in names

    def test_none_existing_score(self):
        fresh = {"summary_scores": [], "traces": [_trace("0")]}
        merged, _ = merge_scores_step_forward(None, fresh)
        assert len(merged["traces"]) == 1
