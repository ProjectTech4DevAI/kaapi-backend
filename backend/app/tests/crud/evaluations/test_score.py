"""`verdict_from_score` — the per-metric verdict band on v2 judge runs.

Bands are upper-bound exclusive: [0, 0.3) Needs Improvement, [0.3, 0.6) Needs
Refinement, [0.6, 1] Good. The enum serializes by value into the trace JSON, so the
display string is part of the contract.
"""

import pytest

from app.crud.evaluations.score import VerdictEnum, verdict_from_score


class TestVerdictFromScore:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.0, VerdictEnum.NEEDS_IMPROVEMENT),
            (0.29, VerdictEnum.NEEDS_IMPROVEMENT),
            (0.3, VerdictEnum.NEEDS_REFINEMENT),
            (0.59, VerdictEnum.NEEDS_REFINEMENT),
            (0.6, VerdictEnum.GOOD),
            (1.0, VerdictEnum.GOOD),
        ],
    )
    def test_bands_and_boundaries(self, score: float, expected: VerdictEnum) -> None:
        assert verdict_from_score(score) is expected

    @pytest.mark.parametrize(
        ("member", "display"),
        [
            (VerdictEnum.NEEDS_IMPROVEMENT, "Needs Improvement"),
            (VerdictEnum.NEEDS_REFINEMENT, "Needs Refinement"),
            (VerdictEnum.GOOD, "Good"),
        ],
    )
    def test_display_string_is_the_serialized_value(
        self, member: VerdictEnum, display: str
    ) -> None:
        assert member.value == display

    def test_returns_verdict_enum_instance(self) -> None:
        assert isinstance(verdict_from_score(0.5), VerdictEnum)
