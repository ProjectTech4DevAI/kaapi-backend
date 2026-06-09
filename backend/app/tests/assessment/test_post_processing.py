"""Tests for the assessment export post-processing engine."""

from app.services.assessment.utils.post_processing import (
    apply_computed_columns,
    apply_filter,
    apply_post_processing,
    apply_sort,
    evaluate_formula,
)


class TestEvaluateFormula:
    def test_addition(self) -> None:
        assert evaluate_formula("@a + @b", {"a": 2, "b": 3}) == 5.0

    def test_all_operators(self) -> None:
        row = {"a": 10, "b": 4}
        assert evaluate_formula("@a - @b", row) == 6.0
        assert evaluate_formula("@a * @b", row) == 40.0
        assert evaluate_formula("@a / @b", row) == 2.5
        assert evaluate_formula("-@a", row) == -10.0

    def test_precedence_and_constants(self) -> None:
        assert evaluate_formula("@a + @b * 0.5", {"a": 1, "b": 4}) == 3.0

    def test_string_numeric_values_coerced(self) -> None:
        assert evaluate_formula("@a + @b", {"a": "2", "b": "3"}) == 5.0

    def test_missing_column_is_zero(self) -> None:
        assert evaluate_formula("@a + @b", {"a": 5}) == 5.0

    def test_non_numeric_value_is_zero(self) -> None:
        assert evaluate_formula("@a + @b", {"a": 5, "b": "abc"}) == 5.0

    def test_unsupported_operation_returns_none(self) -> None:
        # Power operator is not in the safe-ops allowlist.
        assert evaluate_formula("@a ** @b", {"a": 2, "b": 3}) is None

    def test_syntax_error_returns_none(self) -> None:
        assert evaluate_formula("@a +", {"a": 1}) is None


class TestApplyComputedColumns:
    def test_adds_column_in_place(self) -> None:
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        apply_computed_columns(rows, [{"name": "total", "formula": "@a + @b"}])
        assert rows[0]["total"] == 3.0
        assert rows[1]["total"] == 7.0

    def test_skips_empty_name_or_formula(self) -> None:
        rows = [{"a": 1}]
        apply_computed_columns(
            rows,
            [
                {"name": "", "formula": "@a"},
                {"name": "x", "formula": ""},
            ],
        )
        assert rows[0] == {"a": 1}


class TestApplyFilter:
    def test_no_rules_returns_all(self) -> None:
        rows = [{"a": 1}, {"a": 2}]
        assert apply_filter(rows, []) == rows

    def test_eq_ne(self) -> None:
        rows = [{"x": "Yes"}, {"x": "no"}]
        assert apply_filter(rows, [{"column": "x", "op": "eq", "value": "yes"}]) == [
            {"x": "Yes"}
        ]
        assert apply_filter(rows, [{"column": "x", "op": "ne", "value": "yes"}]) == [
            {"x": "no"}
        ]

    def test_contains_not_contains(self) -> None:
        rows = [{"x": "hello world"}, {"x": "bye"}]
        assert apply_filter(
            rows, [{"column": "x", "op": "contains", "value": "world"}]
        ) == [{"x": "hello world"}]
        assert apply_filter(
            rows, [{"column": "x", "op": "not_contains", "value": "world"}]
        ) == [{"x": "bye"}]

    def test_in_not_in(self) -> None:
        rows = [{"x": "a"}, {"x": "b"}]
        assert apply_filter(
            rows, [{"column": "x", "op": "in", "value": ["a", "c"]}]
        ) == [{"x": "a"}]
        assert apply_filter(
            rows, [{"column": "x", "op": "not_in", "value": ["a", "c"]}]
        ) == [{"x": "b"}]

    def test_is_empty_is_not_empty(self) -> None:
        rows = [{"x": ""}, {"x": "v"}, {"x": None}]
        assert apply_filter(rows, [{"column": "x", "op": "is_empty"}]) == [
            {"x": ""},
            {"x": None},
        ]
        assert apply_filter(rows, [{"column": "x", "op": "is_not_empty"}]) == [
            {"x": "v"}
        ]

    def test_numeric_comparisons(self) -> None:
        rows = [{"n": 1}, {"n": 5}, {"n": 10}]
        assert apply_filter(rows, [{"column": "n", "op": "gt", "value": 4}]) == [
            {"n": 5},
            {"n": 10},
        ]
        assert apply_filter(rows, [{"column": "n", "op": "lt", "value": 5}]) == [
            {"n": 1}
        ]
        assert apply_filter(rows, [{"column": "n", "op": "gte", "value": 5}]) == [
            {"n": 5},
            {"n": 10},
        ]
        assert apply_filter(rows, [{"column": "n", "op": "lte", "value": 5}]) == [
            {"n": 1},
            {"n": 5},
        ]

    def test_numeric_filter_non_numeric_excluded(self) -> None:
        rows = [{"n": "abc"}, {"n": 5}]
        assert apply_filter(rows, [{"column": "n", "op": "gt", "value": 1}]) == [
            {"n": 5}
        ]

    def test_unknown_op_keeps_row(self) -> None:
        rows = [{"x": "a"}]
        assert apply_filter(rows, [{"column": "x", "op": "weird", "value": 1}]) == rows

    def test_and_logic_across_rules(self) -> None:
        rows = [{"n": 5, "x": "yes"}, {"n": 5, "x": "no"}, {"n": 1, "x": "yes"}]
        out = apply_filter(
            rows,
            [
                {"column": "n", "op": "gte", "value": 5},
                {"column": "x", "op": "eq", "value": "yes"},
            ],
        )
        assert out == [{"n": 5, "x": "yes"}]


class TestApplySort:
    def test_no_rules_returns_input(self) -> None:
        rows = [{"n": 2}, {"n": 1}]
        assert apply_sort(rows, []) == rows

    def test_numeric_asc_desc(self) -> None:
        rows = [{"n": 3}, {"n": 1}, {"n": 2}]
        assert [
            r["n"] for r in apply_sort(rows, [{"column": "n", "direction": "asc"}])
        ] == [1, 2, 3]
        assert [
            r["n"] for r in apply_sort(rows, [{"column": "n", "direction": "desc"}])
        ] == [3, 2, 1]

    def test_none_values_sort_last(self) -> None:
        rows = [{"n": None}, {"n": 2}, {"n": 1}]
        assert [
            r["n"] for r in apply_sort(rows, [{"column": "n", "direction": "asc"}])
        ] == [1, 2, None]

    def test_string_asc_desc(self) -> None:
        rows = [{"s": "banana"}, {"s": "apple"}, {"s": "cherry"}]
        assert [
            r["s"] for r in apply_sort(rows, [{"column": "s", "direction": "asc"}])
        ] == ["apple", "banana", "cherry"]
        assert [
            r["s"] for r in apply_sort(rows, [{"column": "s", "direction": "desc"}])
        ] == ["cherry", "banana", "apple"]

    def test_multi_rule_priority(self) -> None:
        rows = [
            {"grp": "a", "n": 2},
            {"grp": "b", "n": 1},
            {"grp": "a", "n": 1},
        ]
        out = apply_sort(
            rows,
            [
                {"column": "grp", "direction": "asc"},
                {"column": "n", "direction": "desc"},
            ],
        )
        assert out == [
            {"grp": "a", "n": 2},
            {"grp": "a", "n": 1},
            {"grp": "b", "n": 1},
        ]


class TestApplyPostProcessing:
    def test_none_config_is_noop(self) -> None:
        rows = [{"a": 1}]
        assert apply_post_processing(rows, None) is rows

    def test_full_pipeline(self) -> None:
        rows = [
            {"Novelty": 3, "Feasibility": 4},
            {"Novelty": 9, "Feasibility": 8},
            {"Novelty": 1, "Feasibility": 1},
        ]
        config = {
            "computed_columns": [
                {"name": "Total", "formula": "@Novelty + @Feasibility"}
            ],
            "filter": [{"column": "Total", "op": "gt", "value": 5}],
            "sort": [{"column": "Total", "direction": "desc"}],
        }
        out = apply_post_processing(rows, config)
        assert [r["Total"] for r in out] == [17.0, 7.0]
