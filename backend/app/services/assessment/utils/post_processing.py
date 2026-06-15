"""Post-processing engine for assessment exports.
"""

import ast
import logging
import operator
import re
from typing import Any

logger = logging.getLogger(__name__)

# Safe formula evaluator
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported operation in formula: {ast.dump(node)}")


def evaluate_formula(formula: str, row: dict[str, Any]) -> float | None:
    """Evaluate a formula like '@Novelty_score + @Feasibility_score * 0.5'.

    Returns None if the formula fails or references missing columns.
    """

    def resolve(match: re.Match) -> str:
        col = match.group(1)
        val = row.get(col)
        if val is None:
            return "0"
        try:
            return str(float(val))
        except (TypeError, ValueError):
            return "0"

    expr = re.sub(r"@([\w]+)", resolve, formula)

    try:
        tree = ast.parse(expr, mode="eval")
        return _eval_node(tree.body)
    except Exception as exc:
        logger.warning("[evaluate_formula] Failed to evaluate %r: %s", formula, exc)
        return None


# Filter

_FILTER_OPS = {
    "eq": lambda a, b: str(a).strip().lower() == str(b).strip().lower(),
    "ne": lambda a, b: str(a).strip().lower() != str(b).strip().lower(),
    "contains": lambda a, b: str(b).lower() in str(a).lower(),
    "not_contains": lambda a, b: str(b).lower() not in str(a).lower(),
    "in": lambda a, b: str(a).strip().lower() in {str(v).lower() for v in b},
    "not_in": lambda a, b: str(a).strip().lower() not in {str(v).lower() for v in b},
    "is_empty": lambda a, _: a is None or str(a).strip() == "",
    "is_not_empty": lambda a, _: a is not None and str(a).strip() != "",
}


def _numeric_filter(op: str, a: Any, b: Any) -> bool:
    try:
        fa, fb = float(a), float(b)
        if op == "gt":
            return fa > fb
        if op == "lt":
            return fa < fb
        if op == "gte":
            return fa >= fb
        if op == "lte":
            return fa <= fb
    except (TypeError, ValueError):
        pass
    return False


def _row_matches_filter(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    col = rule["column"]
    op = rule["op"]
    value = rule.get("value")
    cell = row.get(col)

    if op in ("gt", "lt", "gte", "lte"):
        return _numeric_filter(op, cell, value)
    if op in _FILTER_OPS:
        return _FILTER_OPS[op](cell, value)
    return True


def apply_computed_columns(
    rows: list[dict[str, Any]],
    computed_columns: list[dict[str, Any]],
) -> None:
    """Add computed columns to each row in-place."""
    for row in rows:
        for col_def in computed_columns:
            name = col_def.get("name", "").strip()
            formula = col_def.get("formula", "").strip()
            if not name or not formula:
                continue
            row[name] = evaluate_formula(formula, row)


def apply_filter(
    rows: list[dict[str, Any]],
    filter_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only rows that match ALL filter rules (AND logic)."""
    if not filter_rules:
        return rows
    return [
        row
        for row in rows
        if all(_row_matches_filter(row, rule) for rule in filter_rules)
    ]


def apply_sort(
    rows: list[dict[str, Any]],
    sort_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort rows by priority-ordered rules. First rule has highest priority."""
    if not sort_rules:
        return rows

    # Build sort key: iterate rules in reverse (lowest priority first)
    # so that highest priority rule is the final (dominant) tiebreaker.
    result = rows
    for rule in reversed(sort_rules):
        col = rule.get("column", "")
        desc = str(rule.get("direction", "asc")).lower() == "desc"

        def sort_key(row: dict[str, Any], _col: str = col) -> tuple:
            val = row.get(_col)
            if val is None:
                return (1, 0, "")
            try:
                return (0, -float(val) if desc else float(val), "")
            except (TypeError, ValueError):
                s = str(val).lower()
                return (
                    (0, 0, s)
                    if not desc
                    else (0, 0, "".join(chr(0x10FFFF - ord(c)) for c in s))
                )

        result = sorted(result, key=sort_key)

    return result


def apply_post_processing(
    rows: list[dict[str, Any]],
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Apply full post-processing pipeline: computed columns → filter → sort.

    Safe to call with config=None (no-op).
    """
    if not config:
        return rows

    computed_columns = config.get("computed_columns") or []
    filter_rules = config.get("filter") or []
    sort_rules = config.get("sort") or []

    if computed_columns:
        apply_computed_columns(rows, computed_columns)

    rows = apply_filter(rows, filter_rules)
    rows = apply_sort(rows, sort_rules)

    return rows
