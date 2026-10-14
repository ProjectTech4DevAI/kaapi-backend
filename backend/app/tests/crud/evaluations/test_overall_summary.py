"""`compute_overall_summary` — the deterministic run-level weighted overall.

Weights track the real registry (`{key: spec.weight}`) rather than hard-coded
numbers, so a registry reweighting reshapes these expectations instead of silently
passing. Expected overall scores / deltas are hand-computed from the SRD split
(GT 0.5, KB 0.3, prompt 0.2), independent of the implementation's arithmetic.
"""

import pytest

from app.crud.evaluations.judge import METRIC_REGISTRY
from app.crud.evaluations.score import (
    GROUND_TRUTH_SCORE_NAME,
    KNOWLEDGE_BASE_SCORE_NAME,
    PROMPT_SCORE_NAME,
    compute_overall_summary,
    verdict_from_score,
)

METRIC_WEIGHTS = {key.value: spec.weight for key, spec in METRIC_REGISTRY.items()}
METRIC_NAMES = {key.value: spec.score_name for key, spec in METRIC_REGISTRY.items()}


def _summary(metric_avgs: dict[str, float]):
    return compute_overall_summary(
        metric_avgs=metric_avgs,
        metric_weights=METRIC_WEIGHTS,
        metric_names=METRIC_NAMES,
    )


class TestAllThreeMetrics:
    def test_weighted_overall_and_per_dimension_breakdown(self) -> None:
        # 4*0.5 + 3*0.3 + 2*0.2 = 2.0 + 0.9 + 0.4 = 3.3.
        result = _summary({"ground_truth": 4, "knowledge_base": 3, "prompt": 2})
        assert result is not None
        assert result["overall_score"] == 3.3
        assert result["verdict"] == "Needs Refinement"
        assert result["ai_summary"] is None

        by_key = {dim["key"]: dim for dim in result["breakdown"]}
        assert by_key["ground_truth"]["name"] == GROUND_TRUTH_SCORE_NAME
        assert by_key["knowledge_base"]["name"] == KNOWLEDGE_BASE_SCORE_NAME
        assert by_key["prompt"]["name"] == PROMPT_SCORE_NAME

        # All three present → base weights already sum to 1, so no renormalization.
        assert by_key["ground_truth"]["weight"] == 0.5
        assert by_key["knowledge_base"]["weight"] == 0.3
        assert by_key["prompt"]["weight"] == 0.2
        assert sum(dim["weight"] for dim in result["breakdown"]) == 1.0

        assert by_key["ground_truth"]["score"] == 4
        assert by_key["knowledge_base"]["score"] == 3
        assert by_key["prompt"]["score"] == 2

        # delta = dimension score - overall (3.3), sign shows pull vs the run.
        assert by_key["ground_truth"]["delta"] == 0.7
        assert by_key["knowledge_base"]["delta"] == -0.3
        assert by_key["prompt"]["delta"] == -1.3

        assert by_key["ground_truth"]["verdict"] == "Good"
        assert by_key["knowledge_base"]["verdict"] == "Needs Refinement"
        assert by_key["prompt"]["verdict"] == "Needs Refinement"

    def test_per_dimension_verdict_matches_the_rounded_dimension_score(self) -> None:
        result = _summary({"ground_truth": 2.75, "knowledge_base": 1.25, "prompt": 4.5})
        assert result is not None
        for dim in result["breakdown"]:
            assert dim["verdict"] == verdict_from_score(dim["score"]).value


class TestRenormalizationWhenAMetricIsAbsent:
    def test_ground_truth_and_knowledge_base_only(self) -> None:
        result = _summary({"ground_truth": 4, "knowledge_base": 3})
        assert result is not None
        by_key = {dim["key"]: dim for dim in result["breakdown"]}
        assert set(by_key) == {"ground_truth", "knowledge_base"}

        # 0.5 and 0.3 renormalize over 0.8 → 0.625 and 0.375. The stored weights are
        # rounded to 2 dp; 0.3/0.8 floats to 0.3749… so it rounds DOWN to 0.37, and
        # the displayed pair sums to 0.99 (the un-rounded renorm that drives the
        # overall still sums to 1.0).
        assert by_key["ground_truth"]["weight"] == 0.62
        assert by_key["knowledge_base"]["weight"] == 0.37
        # 0.625*4 + 0.375*3 = 2.5 + 1.125 = 3.625 → 3.62.
        assert result["overall_score"] == 3.62

    def test_ground_truth_and_prompt_only(self) -> None:
        result = _summary({"ground_truth": 4, "prompt": 2})
        assert result is not None
        by_key = {dim["key"]: dim for dim in result["breakdown"]}
        assert set(by_key) == {"ground_truth", "prompt"}

        # 0.5 and 0.2 renormalize over 0.7 → 0.714… and 0.285… → 0.71 and 0.29.
        assert by_key["ground_truth"]["weight"] == 0.71
        assert by_key["prompt"]["weight"] == 0.29
        assert sum(dim["weight"] for dim in result["breakdown"]) == 1.0

    def test_absent_metric_never_drags_the_overall_down(self) -> None:
        # A missing metric is dropped, not scored 0: two perfect metrics stay at 5.0.
        result = _summary({"ground_truth": 5, "knowledge_base": 5})
        assert result is not None
        assert result["overall_score"] == 5.0


class TestNothingScored:
    def test_empty_metric_avgs_returns_none(self) -> None:
        assert _summary({}) is None


class TestBadgeBoundaries:
    def test_overall_exactly_on_needs_refinement_lower_bound(self) -> None:
        # Equal dimension averages → the weighted overall equals that value exactly.
        result = _summary({"ground_truth": 2, "knowledge_base": 2, "prompt": 2})
        assert result is not None
        assert result["overall_score"] == 2.0
        assert result["verdict"] == "Needs Refinement"
        for dim in result["breakdown"]:
            assert dim["verdict"] == "Needs Refinement"

    def test_overall_exactly_on_good_lower_bound(self) -> None:
        result = _summary({"ground_truth": 4, "knowledge_base": 4, "prompt": 4})
        assert result is not None
        assert result["overall_score"] == 4.0
        assert result["verdict"] == "Good"
        for dim in result["breakdown"]:
            assert dim["verdict"] == "Good"
