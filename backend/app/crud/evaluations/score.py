"""
Type definitions for evaluation scores.

This module contains TypedDict definitions for type-safe score data
used throughout the evaluation system.
"""

from typing import NotRequired, TypedDict


class TraceScore(TypedDict):
    """A score attached to a trace."""

    name: str
    value: float | str
    data_type: str
    comment: NotRequired[str]


class TraceData(TypedDict):
    """Data for a single trace including Q&A and scores."""

    trace_id: str
    question: str
    llm_answer: str
    ground_truth_answer: str
    scores: list[TraceScore]


class NumericSummaryScore(TypedDict):
    """Summary statistics for a numeric score across all traces."""

    name: str
    avg: float
    std: float
    total_pairs: int
    data_type: str


class CategoricalSummaryScore(TypedDict):
    """Summary statistics for a categorical score across all traces."""

    name: str
    distribution: dict[str, int]
    total_pairs: int
    data_type: str


SummaryScore = NumericSummaryScore | CategoricalSummaryScore


class EvaluationScore(TypedDict):
    """Complete evaluation score data with traces and summary statistics."""

    summary_scores: list[SummaryScore]
    traces: list[TraceData]
