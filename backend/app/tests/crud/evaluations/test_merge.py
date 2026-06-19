"""Tests for step-forward trace merge helpers used on Langfuse resync."""

from app.crud.evaluations.merge import (
    compute_summary_scores,
    merge_scores_step_forward,
    merge_trace_data,
    sort_traces_by_question_id,
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

    def test_category_and_external_id_preserved_across_resync(self):
        def _real(trace_id, value=0.5, category="Health"):
            return {
                **_trace(trace_id, value=value),
                "category": category,
            }

        # Both sides have the keys → merged carries them.
        existing = [_real("1", value=0.5, category="Health")]
        fresh = [_real("1", value=0.9, category="Health")]
        merged, stats = merge_trace_data(existing, fresh)
        assert merged[0]["category"] == "Health"
        assert merged[0]["scores"][0]["value"] == 0.9
        assert stats["updated"] == 1

        # Existing has them, fresh missing the keys → fall back to existing.
        existing2 = [_real("2", category="Education")]
        fresh2 = [_trace("2", value=1.0)]  # legacy-shape trace
        merged2, _ = merge_trace_data(existing2, fresh2)
        assert merged2[0]["category"] == "Education"

        # Fresh has them, existing missing the keys → take from fresh.
        existing3 = [_trace("3")]  # legacy-shape trace
        fresh3 = [_real("3", category="Sports")]
        merged3, _ = merge_trace_data(existing3, fresh3)
        assert merged3[0]["category"] == "Sports"


class TestSortTracesByQuestionId:
    """Sort traces by the upload-time question_id so the API response follows the
    CSV's natural order instead of ThreadPoolExecutor completion order."""

    def test_sorts_int_question_ids_ascending(self):
        traces = [
            {"trace_id": "a", "question_id": 3},
            {"trace_id": "b", "question_id": 1},
            {"trace_id": "c", "question_id": 2},
        ]
        sorted_traces = sort_traces_by_question_id(traces)
        assert [t["question_id"] for t in sorted_traces] == [1, 2, 3]

    def test_sorts_string_digit_question_ids_numerically(self):
        """Numeric strings sort like integers, not lexicographically: '10' > '2'."""
        traces = [
            {"trace_id": "a", "question_id": "10"},
            {"trace_id": "b", "question_id": "2"},
            {"trace_id": "c", "question_id": "1"},
        ]
        sorted_traces = sort_traces_by_question_id(traces)
        assert [t["question_id"] for t in sorted_traces] == ["1", "2", "10"]

    def test_missing_or_non_numeric_question_id_pushed_to_end(self):
        """Legacy traces without question_id sort after the numbered ones so the
        sort stays total — exercises the fallback branch in the sort key.
        """
        traces = [
            {"trace_id": "a", "question_id": 2},
            {"trace_id": "b"},  # missing
            {"trace_id": "c", "question_id": ""},  # empty string
            {"trace_id": "d", "question_id": "abc"},  # non-numeric
            {"trace_id": "e", "question_id": 1},
        ]
        sorted_traces = sort_traces_by_question_id(traces)
        assert [t["trace_id"] for t in sorted_traces[:2]] == ["e", "a"]
        # Order among the missing/non-numeric tail is stable but not specified.
        assert {t["trace_id"] for t in sorted_traces[2:]} == {"b", "c", "d"}

    def test_merge_scores_step_forward_returns_sorted_traces(self):
        """The merge result is the chokepoint that gets persisted, so it must
        come back ordered by question_id regardless of input order.
        """
        cached = {
            "summary_scores": [],
            "traces": [_trace(str(i)) for i in (2, 0, 1)],
        }
        fresh = {
            "summary_scores": [],
            "traces": [_trace(str(i)) for i in (4, 3)],
        }
        merged, _ = merge_scores_step_forward(cached, fresh)
        assert [t["question_id"] for t in merged["traces"]] == [0, 1, 2, 3, 4]


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


def _cosine_trace(trace_id, value=None, *, unscoreable=False, reason="empty_output"):
    """A trace carrying (or not) a Cosine Similarity score."""
    scores = []
    if value is not None:
        scores.append(
            {"name": "Cosine Similarity", "value": value, "data_type": "NUMERIC"}
        )
    if unscoreable:
        scores.append(
            {
                "name": "Cosine Similarity",
                "value": 0,
                "data_type": "NUMERIC",
                "comment": f"Cannot compute: {reason}",
                "unscoreable": True,
            }
        )
    return {
        "trace_id": trace_id,
        "question": f"q{trace_id}",
        "llm_answer": "a",
        "ground_truth_answer": "g",
        "question_id": int(trace_id),
        "scores": scores,
    }


class TestComputeSummaryScoresUnscoreable:
    def test_unscoreable_excluded_from_stats(self):
        """A 0-value unscoreable entry never enters avg/std/total_pairs."""
        traces = [
            _cosine_trace("0", value=0.8),
            _cosine_trace("1", value=0.6),
            _cosine_trace("2", unscoreable=True),
        ]
        summary = compute_summary_scores(traces)
        cosine = next(s for s in summary if s["name"] == "Cosine Similarity")
        assert cosine["total_pairs"] == 2
        assert cosine["avg"] == 0.7  # mean(0.8, 0.6), not dragged down by the 0


class TestBackfillMissingScores:
    def test_injects_missing_score(self):
        from app.crud.evaluations.merge import backfill_missing_scores

        traces = [_cosine_trace("0", value=0.8), _cosine_trace("1")]  # trace 1 empty
        backfill_missing_scores(traces, {"1": 0.55})
        summary = compute_summary_scores(traces)
        cosine = next(s for s in summary if s["name"] == "Cosine Similarity")
        assert cosine["total_pairs"] == 2

    def test_does_not_overwrite_existing(self):
        from app.crud.evaluations.merge import backfill_missing_scores

        traces = [_cosine_trace("0", value=0.8)]
        backfill_missing_scores(traces, {"0": 0.1})
        # Existing 0.8 kept; no duplicate cosine entry appended.
        cosine_scores = [
            s for s in traces[0]["scores"] if s["name"] == "Cosine Similarity"
        ]
        assert len(cosine_scores) == 1
        assert cosine_scores[0]["value"] == 0.8

    def test_unknown_trace_ids_ignored(self):
        from app.crud.evaluations.merge import backfill_missing_scores

        traces = [_cosine_trace("0")]
        backfill_missing_scores(traces, {"99": 0.5})
        assert traces[0]["scores"] == []

    def test_none_map_is_noop(self):
        from app.crud.evaluations.merge import backfill_missing_scores

        traces = [_cosine_trace("0")]
        backfill_missing_scores(traces, None)
        assert traces[0]["scores"] == []


class TestApplyCosineBreakdown:
    def test_adds_total_items_and_unscoreable(self):
        from app.crud.evaluations.merge import apply_cosine_breakdown

        summary = [
            {
                "name": "Cosine Similarity",
                "avg": 0.7,
                "std": 0.1,
                "total_pairs": 2,
                "data_type": "NUMERIC",
            }
        ]
        apply_cosine_breakdown(
            summary,
            total_items=4,
            unscoreable={
                "t1": "empty_output",
                "t2": "empty_output",
                "t3": "embedding_failed",
            },
        )
        cosine = summary[0]
        assert cosine["total_items"] == 4
        assert cosine["unscoreable"] == {"empty_output": 2, "embedding_failed": 1}

    def test_noop_without_cosine_entry(self):
        from app.crud.evaluations.merge import apply_cosine_breakdown

        summary = [{"name": "accuracy", "total_pairs": 1, "data_type": "NUMERIC"}]
        apply_cosine_breakdown(
            summary, total_items=10, unscoreable={"t": "empty_output"}
        )
        assert "total_items" not in summary[0]


class TestMergeBackfillIntegration:
    def test_resync_backfills_computed_but_unwritten_scores(self):
        """The Langfuse fetch is missing a score that we computed; backfill recovers it.

        Reproduces the 211/215 freeze: fresh Langfuse traces have scores for
        only some items, but per_item_scores (durable source of truth) has all.
        """
        # Cached has both traces but no cosine scores yet (first batch trace view).
        cached = {
            "summary_scores": [],
            "traces": [_cosine_trace("0"), _cosine_trace("1")],
        }
        # Langfuse only returned a score for trace 0 (trace 1's write was lost).
        fresh = {"summary_scores": [], "traces": [_cosine_trace("0", value=0.9)]}

        merged, _ = merge_scores_step_forward(
            cached, fresh, per_item_scores={"0": 0.9, "1": 0.4}
        )
        cosine = next(
            s for s in merged["summary_scores"] if s["name"] == "Cosine Similarity"
        )
        assert cosine["total_pairs"] == 2  # both recovered, not 1
