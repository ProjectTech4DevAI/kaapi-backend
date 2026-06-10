"""Unit tests for `trace_sort_key` — the ordering helper used by both
row- and grouped-format eval responses."""

from app.crud.evaluations.score import trace_sort_key


class TestTraceSortKey:
    def test_numeric_external_ids_sort_by_value(self):
        traces = [
            {"external_id": "10", "question_id": 3},
            {"external_id": "2", "question_id": 1},
            {"external_id": "1", "question_id": 2},
        ]
        traces.sort(key=trace_sort_key)
        assert [t["external_id"] for t in traces] == ["1", "2", "10"]

    def test_non_numeric_external_ids_sort_lexicographically_after_numerics(self):
        traces = [
            {"external_id": "zebra", "question_id": 1},
            {"external_id": "2", "question_id": 2},
            {"external_id": "alpha", "question_id": 3},
            {"external_id": "1", "question_id": 4},
        ]
        traces.sort(key=trace_sort_key)
        assert [t["external_id"] for t in traces] == ["1", "2", "alpha", "zebra"]

    def test_missing_external_id_falls_back_to_question_id(self):
        traces = [
            {"external_id": "5", "question_id": 1},
            {"external_id": None, "question_id": 2},
            {"external_id": "1", "question_id": 3},
            {"external_id": None, "question_id": 4},
        ]
        traces.sort(key=trace_sort_key)
        order = [(t["external_id"], t["question_id"]) for t in traces]
        # externals first sorted by id, then question_id fallbacks in qid order
        assert order == [("1", 3), ("5", 1), (None, 2), (None, 4)]

    def test_all_legacy_traces_sort_by_question_id(self):
        traces = [
            {"external_id": None, "question_id": 3},
            {"external_id": None, "question_id": 1},
            {"external_id": None, "question_id": 2},
        ]
        traces.sort(key=trace_sort_key)
        assert [t["question_id"] for t in traces] == [1, 2, 3]

    def test_empty_string_external_id_treated_as_missing(self):
        traces = [
            {"external_id": "", "question_id": 2},
            {"external_id": "5", "question_id": 1},
        ]
        traces.sort(key=trace_sort_key)
        assert traces[0]["external_id"] == "5"
        assert traces[1]["external_id"] == ""

    def test_whitespace_padded_id_normalized(self):
        traces = [
            {"external_id": "  3  ", "question_id": 1},
            {"external_id": "1", "question_id": 2},
        ]
        traces.sort(key=trace_sort_key)
        assert traces[0]["external_id"] == "1"
        assert traces[1]["external_id"].strip() == "3"

    def test_duplicate_external_ids_stable_sort_preserves_input_order(self):
        traces = [
            {"external_id": "1", "question_id": 5, "marker": "A"},
            {"external_id": "1", "question_id": 5, "marker": "B"},
            {"external_id": "1", "question_id": 5, "marker": "C"},
        ]
        traces.sort(key=trace_sort_key)
        assert [t["marker"] for t in traces] == ["A", "B", "C"]

    def test_negative_external_ids_sort_numerically(self):
        traces = [
            {"external_id": "-3", "question_id": 1},
            {"external_id": "1", "question_id": 2},
            {"external_id": "-1", "question_id": 3},
        ]
        traces.sort(key=trace_sort_key)
        assert [t["external_id"] for t in traces] == ["-3", "-1", "1"]

    def test_question_id_fallback_with_non_int_question_id(self):
        # Defensive: if question_id is a string (legacy / edge case), key
        # still produces a stable tuple — no TypeError on sort.
        traces = [
            {"external_id": None, "question_id": "bbb"},
            {"external_id": None, "question_id": "aaa"},
        ]
        traces.sort(key=trace_sort_key)
        assert [t["question_id"] for t in traces] == ["aaa", "bbb"]

    def test_missing_both_external_id_and_question_id(self):
        # Trace with neither field still sorts deterministically (key returns
        # a fixed tuple), doesn't raise.
        traces = [
            {},
            {"external_id": "1"},
        ]
        traces.sort(key=trace_sort_key)
        # external_id="1" should come before the empty trace
        assert traces[0].get("external_id") == "1"
